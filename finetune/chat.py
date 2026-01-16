import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "./chatbot/merge_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')
model.eval()

def chat(history, max_new_tokens=128):
    """
    history: List[str]，按顺序存放对话
    """
    # BlenderBot 使用简单拼接即可
    prompt = "\n".join(history)

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )
    return response.strip()

# ===== 开始对话 =====
history = []

print('all ready')

while True:
    history.append(f"human: {input()}")
    reply = chat(history)
    print("Bot:", reply)
    history.append(f"Bot: {reply}")

