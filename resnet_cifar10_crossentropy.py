
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 适配cifar10
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]

# 为训练集定义数据增强和转换
transform_train = transforms.Compose([
    # 随机调节亮度、对比度、饱和度和色相
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# 下载并加载CIFAR-10数据集
train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)

# 创建数据加载器
batch_size = 128
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# CIFAR-10的类别
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# 3. 定义ResNet-50模型
# 加载预训练的ResNet-50模型
# pretrained=True 会下载在ImageNet上预训练的权重
# 这里我们设置为False，从头开始训练，当然也可以设置为True进行微调
model = resnet50(weights=None)


num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

# sgd里面设置智能冲量
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True, weight_decay=5e-4)

# 间隔7个周期，依次将学历率乘0.1
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# 5. 训练模型
def train(epoch):
    print(f'\nEpoch: {epoch + 1}')
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # 将数据移动到指定设备
        inputs, targets = inputs.to(device), targets.to(device)

        # 梯度清零
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)
        
        # 计算损失
        loss = criterion(outputs, targets)
        
        # 反向传播
        loss.backward()
        
        # 更新权重
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_idx + 1) % 100 == 0:
            print(f'  Batch [{batch_idx + 1}/{len(train_loader)}] | ' 
                  f'Loss: {running_loss / (batch_idx + 1):.3f} | ' 
                  f'Acc: {100. * correct / total:.3f}% ({correct}/{total})')

def test():
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print(f'\nTest Results:')
    print(f'  Loss: {test_loss / len(test_loader):.3f}')
    print(f'  Accuracy: {100. * correct / total:.3f}% ({correct}/{total})')


if __name__ == '__main__':
    num_epochs = 100

    for epoch in range(num_epochs):
        train(epoch)
        test()
        scheduler.step()
    # torch.save(model.state_dict(), 'resnet50_cifar10.pth')
