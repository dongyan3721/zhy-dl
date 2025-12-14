import os
import re

import torch
from PIL import Image
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms


class Residual(nn.Module):  #@save
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            # 第一层进来做1*1卷积，因为特征已经被降了很多
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X # res
        return F.relu(Y)

b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

b2 = resnet_block(64, 64, 2, True)
b3 = resnet_block(64, 128, 2)
b4 = resnet_block(128, 256, 2)
b5 = resnet_block(256, 512, 2)

new_model = nn.Sequential(b1, nn.Sequential(*b2), nn.Sequential(*b3), nn.Sequential(*b4), nn.Sequential(*b5),
                        # 自适应卷积，前两维1*1
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(), nn.Linear(512, 2))
new_model.load_state_dict(torch.load("res_net.pth"))

class CatDataset(Dataset):
    def __init__(self, image_path, transform=None, is_train=True):
        self.paths = []
        self.labels = []
        for w, _, files in os.walk(image_path):
            for f in files:
                if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg'):
                    self.paths.append(os.path.join(w, f))
                    # 1是有耄耋面相，0不是耄耋
                    self.labels.append(int(re.findall(r'\d+', f.split('-')[-1])[0]))
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # 读取图像
        image = Image.open(self.paths[idx]).convert('RGB')

        # 应用数据增强
        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx]

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_dataset = CatDataset(
    image_path='./dataset/maodie/validation',
    transform=val_transform,
    is_train=False
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

new_model.to(device)

new_model.eval()
with torch.no_grad():
    for X, y in val_loader:
        X, y = X.to(device), y.to(device)
        output = new_model(X)
        _, predicted = torch.max(output.data, 1)
        compare = predicted.eq(y.data).cpu().numpy()

        for i, (f, p) in enumerate(zip(compare, val_dataset.paths)):
            print(f'预测{i+1}: {p}: 原是耄耋，预测结果为: {'耄耋' if f else '不是耄耋'}')
