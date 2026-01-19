from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from peft import LoraConfig, TaskType, get_peft_model


def process_func(example):
    MAX_LENGTH = 256
    # 千问131072的窗口
    # 这里不会带特殊字符
    instruction = tokenizer("\n".join(["Human: " + example["INSTRUCTION"]]).strip() + "\n\nAssistant: ")
    response = tokenizer(example["RESPONSE"] + tokenizer.eos_token)

    # 拼接完整对话
    input_ids = instruction["input_ids"] + response["input_ids"]
    attention_mask = instruction["attention_mask"] + response["attention_mask"]

    # -100不参与交叉熵计算
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

if __name__ == '__main__':
    ds = Dataset.from_parquet('../dataset/finetune/train-00000-of-00001-bb5f874d67d84fd2.parquet')

    tokenizer = AutoTokenizer.from_pretrained("../models/Qwen/Qwen3-4B")
    model = AutoModelForCausalLM.from_pretrained("../models/Qwen/Qwen3-4B", device_map='auto')

    tokenized_ds = ds.map(process_func, remove_columns=ds.column_names)
    train_test_dict = tokenized_ds.train_test_split(train_size=0.8) # 好像也可以不搞

    # 需要被添加lora旁支的层
    config = LoraConfig(task_type=TaskType.CAUSAL_LM, target_modules=".*self_attn\.\w_proj", r=6)

    model = get_peft_model(model, config)

    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir="./cache",
        per_device_train_batch_size=1,
        # 还是梯度累计加快训练
        gradient_accumulation_steps=8,
        logging_steps=10,
        num_train_epochs=1
    )


    trainer = Trainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        train_dataset=train_test_dict['train'],
        # 单一批量max_l相同，这时候会给解码器部分添加初始的bos来启动推理
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    trainer.train()

