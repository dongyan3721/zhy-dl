import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from torchensemble.fusion import FusionClassifier
from torchensemble.voting import VotingClassifier
from torchensemble.bagging import BaggingClassifier
from torchensemble.gradient_boosting import GradientBoostingClassifier
from torchensemble.snapshot_ensemble import SnapshotEnsembleClassifier
from torchensemble.soft_gradient_boosting import SoftGradientBoostingClassifier

from torchensemble.utils.logging import set_logger


def display_records(records, logger):
    msg = (
        "{:<28} | Testing Acc: {:.2f} % | Training Time: {:.2f} s |"
        " Evaluating Time: {:.2f} s"
    )

    print("\n")
    for method, training_time, evaluating_time, acc in records:
        logger.info(msg.format(method, acc, training_time, evaluating_time))


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 400)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":

    # Hyper-parameters
    n_estimators = 5
    lr = 1e-3
    weight_decay = 5e-4
    epochs = 100

    # Utils
    batch_size = 128
    records = []
    torch.manual_seed(0)

    # Load data
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    # root变量下需要存放cifar-10-python.tar.gz 文件
    # cifar-10-python.tar.gz可从 "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz" 下载
    train_set = datasets.CIFAR10(root='./data', train=True, transform=train_transform, download=True)
    test_set = datasets.CIFAR10(root='./data', train=False, transform=valid_transform, download=True)

    # 构建DataLoder
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(dataset=test_set, batch_size=batch_size, num_workers=4)

    logger = set_logger("classification_cifar10_cnn", use_tb_logger=True)
    #
    # ============================= FusionClassifier =============================
    """
    直接对模型的输出按权重加权融合
    
    """
    model = FusionClassifier(
        estimator=LeNet5, n_estimators=n_estimators, cuda=True
    )

    # Set the optimizer
    model.set_optimizer("Adam", lr=lr, weight_decay=weight_decay)

    # Training
    tic = time.time()
    model.fit(train_loader, epochs=epochs)
    toc = time.time()
    training_time = toc - tic

    # Evaluating
    tic = time.time()
    testing_acc = model.evaluate(valid_loader)
    toc = time.time()
    evaluating_time = toc - tic

    records.append(
        ("FusionClassifier", training_time, evaluating_time, testing_acc)
    )

    # ============================= VotingClassifier =============================
    # 多个模型投票，如
    """
    模型1（ResNet）预测 0.7 概率属于A
    模型2（MLP）预测 0.6 概率属于A
    模型3（CNN）预测 0.9 概率属于A
    这几个概率平均
    """
    model = VotingClassifier(
        estimator=LeNet5, n_estimators=n_estimators, cuda=True
    )

    # Set the optimizer
    model.set_optimizer("Adam", lr=lr, weight_decay=weight_decay)

    # Training
    tic = time.time()
    model.fit(train_loader, epochs=epochs)
    toc = time.time()
    training_time = toc - tic

    # Evaluating
    tic = time.time()
    testing_acc = model.evaluate(valid_loader)
    toc = time.time()
    evaluating_time = toc - tic

    records.append(
        ("VotingClassifier", training_time, evaluating_time, testing_acc)
    )

    # ============================= BaggingClassifier =============================
    """"
    用相同结构的模型训练多个副本，然后平均预测
    """
    model = BaggingClassifier(
        estimator=LeNet5, n_estimators=n_estimators, cuda=True
    )

    # Set the optimizer
    model.set_optimizer("Adam", lr=lr, weight_decay=weight_decay)

    # Training
    tic = time.time()
    model.fit(train_loader, epochs=epochs)
    toc = time.time()
    training_time = toc - tic

    # Evaluating
    tic = time.time()
    testing_acc = model.evaluate(valid_loader)
    toc = time.time()
    evaluating_time = toc - tic

    records.append(
        ("BaggingClassifier", training_time, evaluating_time, testing_acc)
    )

    # ============================= GradientBoostingClassifier =============================
    """"
    一次训练一个模型，每个新模型都尝试纠正之前模型的错误
    第二个学习器学习的是第一个学习器与目标检测的差距,第三个学习器学习的是第一个+第二个学习器结果之和与结果之间的差距,以此类推
    """
    model = GradientBoostingClassifier(
        estimator=LeNet5, n_estimators=n_estimators, cuda=True
    )

    # Set the optimizer
    model.set_optimizer("Adam", lr=lr, weight_decay=weight_decay)

    # Training
    tic = time.time()
    # model.fit(train_loader, epochs=epochs)
    model.fit(train_loader, epochs=1)
    toc = time.time()
    training_time = toc - tic

    # Evaluating
    tic = time.time()
    testing_acc = model.evaluate(valid_loader)
    toc = time.time()
    evaluating_time = toc - tic

    records.append(
        (
            "GradientBoostingClassifier",
            training_time,
            evaluating_time,
            testing_acc,
        )
    )

    # ============================= SnapshotEnsembleClassifier =============================
    """
    在一次训练中保存多个不同阶段的模型权重作为多个模型(只训练一次模型)
    在训练过程中得到多个局部最优模型快照
    将这些快照拼合为最终模型
    """
    model = SnapshotEnsembleClassifier(
        estimator=LeNet5, n_estimators=n_estimators, cuda=True
    )

    # Set the optimizer
    model.set_optimizer("Adam", lr=lr, weight_decay=weight_decay)

    # Training
    tic = time.time()
    model.fit(train_loader, epochs=epochs)
    toc = time.time()
    training_time = toc - tic

    # Evaluating
    tic = time.time()
    testing_acc = model.evaluate(valid_loader)
    toc = time.time()
    evaluating_time = toc - tic

    records.append(
        (
            "SnapshotEnsembleClassifier",
            training_time,
            evaluating_time,
            testing_acc,
        )
    )

    # ============================= SoftGradientBoostingClassifier =============================
    # 平滑版的梯度提升集成法
    model = SoftGradientBoostingClassifier(
        estimator=LeNet5, n_estimators=n_estimators, cuda=True
    )

    # Set the optimizer
    model.set_optimizer("Adam", lr=lr, weight_decay=weight_decay)

    # Training
    tic = time.time()
    model.fit(train_loader, epochs=epochs)
    toc = time.time()
    training_time = toc - tic

    # Evaluating
    tic = time.time()
    testing_acc = model.evaluate(valid_loader)
    toc = time.time()
    evaluating_time = toc - tic

    records.append(
        (
            "SoftGradientBoostingClassifier",
            training_time,
            evaluating_time,
            testing_acc,
        )
    )

    # Print results on different ensemble methods
    display_records(records, logger)