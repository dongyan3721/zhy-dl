from torchmetrics.utilities.enums import ClassificationTask

if __name__ == "__main__":
    from yu_utils import setup_seed
    setup_seed(40)
    import torch
    import torchmetrics

    metric = torchmetrics.Accuracy(task=ClassificationTask.MULTICLASS, num_classes=5) # type: ignore
    n_batches = 3
    for i in range(n_batches):
        # 10*5，列这个维度做 sigmoid
        preds = torch.randn(10, 5).softmax(dim=-1)
        target = torch.randint(5, (10,))
        # __call__方法等价于调用 forward 操作，forward 内部 update 几个指标
        acc = metric(preds, target)  # 单次计算，并记录本次信息。通过维护tp, tn, fp, fn来记录所有数据
        print(f"Accuracy on batch {i}: {acc}")

    # 累计状态得到准确率
    acc_avg = metric.compute()
    print(f"Accuracy on all data: {acc_avg}")
    tp, tn, fp, fn = metric.tp, metric.tn, metric.fp, metric.fn
    print(tp, tn, fp, fn, sum([tp, tn, fp, fn]))
    metric.reset()
