import os.path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

device = torch.device("cuda" if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else "cpu")

param_exist = os.path.exists("resnet50_cifar10.pth")

writer = SummaryWriter(log_dir='./logs')

# 适配cifar10
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]

# 为训练集定义数据增强和转换
transform_train = transforms.Compose([
    # 随机调节亮度、对比度、饱和度和色相
    # transforms.ColorJitter(
    #     brightness=0.2,
    #     contrast=0.2,
    #     saturation=0.2,
    #     hue=0.1
    # ),
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

model = resnet50(weights=None)


num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

model = model.to(device)

loss = nn.CrossEntropyLoss()

# sgd里面设置智能冲量
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True, weight_decay=5e-4)

# 间隔7个周期，依次将学历率乘0.1
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

def visualize_conv_layers(epoch):
    """可视化第一个和第二个卷积层的卷积核"""
    # 可视化第一个卷积层 (conv1)
    conv1_weights = model.conv1.weight.detach().cpu()
    # 改变形状以便于可视化: (out_channels, in_channels, H, W) -> (out_channels, H, W, in_channels)
    # TensorBoard的add_image期望 (C, H, W), 我们将out_channels视为批次大小
    # 我们只可视化RGB通道的权重
    conv1_weights_rgb = conv1_weights
    grid = torchvision.utils.make_grid(conv1_weights_rgb, nrow=8, normalize=True, scale_each=True)
    writer.add_image('Conv1/weights', grid, global_step=epoch)

    # 可视化第二个卷积层 (layer1[0].conv1)
    # 这是ResNet第一个残差块中的第一个卷积层
    conv2_weights = model.layer1[0].conv1.weight.detach().cpu()
    # 这是一个1x1的卷积，我们需要将其reshape以便可视化
    # (out_channels, in_channels, 1, 1) -> (out_channels, 1, sqrt(in_channels), sqrt(in_channels))
    # 为了简化，我们只取第一个输入通道的权重进行可视化
    conv2_weights_to_vis = conv2_weights[:, 0, :, :].unsqueeze(1) # (out_channels, 1, H, W)
    grid = torchvision.utils.make_grid(conv2_weights_to_vis, nrow=8, normalize=True, scale_each=True)
    writer.add_image('Conv2 (layer1[0].conv1)/weights', grid, global_step=epoch)

last_feature_map = None
def hook_fn(module, input, output):
    global last_feature_map
    last_feature_map = output.detach()

# 最后一个卷积层在layer4的最后一个block
# model.layer4[-1].conv3.register_forward_hook(hook_fn)
# 或者，我们可视化进入全连接层之前的特征图
handle = model.avgpool.register_forward_hook(lambda mod, inp, out: hook_fn(mod, inp, inp[0]))


def test():
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    # 参数确定，在这可视化卷积层权重
    visualize_conv_layers(0)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            l = loss(outputs, targets)

            test_loss += l.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if last_feature_map is not None:
                # 我们只取第一张图片的特征图进行可视化
                feature_map_for_one_image = last_feature_map[0]  # Shape: [C, H, W]
                # 将每个通道的特征图分开
                feature_map_unrolled = [feature_map_for_one_image[i].unsqueeze(0) for i in
                                        range(feature_map_for_one_image.shape[0])]
                grid = torchvision.utils.make_grid(feature_map_unrolled, nrow=32, normalize=True, scale_each=True)
                writer.add_image('Last_Feature_Map/epoch_{}'.format(0), grid)

    print(f'\nTest Results:')
    print(f'  Loss: {test_loss / len(test_loader):.3f}')
    print(f'  Accuracy: {100. * correct / total:.3f}% ({correct}/{total})')


if __name__ == '__main__':
    num_epochs = 50

    x = torch.randn(1, 3, 224, 224)
    summary(model, input_size=x.size())

    if param_exist:
        model.load_state_dict(torch.load('resnet50_cifar10.pth'))
        test()
    else:
        for epoch in range(num_epochs):
            print(f'\nEpoch: {epoch + 1}')
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()

                outputs = model(inputs)  # batch, 10
                l = loss(outputs, targets)
                l.backward()
                optimizer.step()

                running_loss += l.item()
                _, predicted = outputs.max(1) # batch,
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                if (batch_idx + 1) % 100 == 0:
                    print(f'  Batch [{batch_idx + 1}/{len(train_loader)}] | '
                          f'Loss: {running_loss / (batch_idx + 1):.3f} | '
                          f'Acc: {100. * correct / total:.3f}% ({correct}/{total})')
            test()
            # 这个地方也更新一下epoch不然学习率调整不生效
            scheduler.step()
    torch.save(model.state_dict(), 'resnet50_cifar10.pth')
