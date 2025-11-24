import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
import albumentations as A
import numpy as np
import cv2
import re

def transform_invert(img_, transform_train):
    """
    将data 进行反transfrom操作
    :param img_: tensor
    :param transform_train: torchvision.transforms
    :return: PIL image
    """
    if 'Normalize' in str(transform_train):
        norm_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform_train.transforms))
        mean = torch.tensor(norm_transform[0].mean, dtype=img_.dtype, device=img_.device)
        std = torch.tensor(norm_transform[0].std, dtype=img_.dtype, device=img_.device)
        img_.mul_(std[:, None, None]).add_(mean[:, None, None])

    img_ = img_.transpose(0, 2).transpose(0, 1)  # C*H*W --> H*W*C
    if 'ToTensor' in str(transform_train) or img_.max() < 1:
        img_ = img_.detach().numpy() * 255

    if img_.shape[2] == 3:
        img_ = Image.fromarray(img_.astype('uint8')).convert('RGB')
    elif img_.shape[2] == 1:
        img_ = Image.fromarray(img_.astype('uint8').squeeze())
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_.shape[2]))

    return img_


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

        img_bgr = cv2.imread(self.paths[idx])
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 应用数据增强
        if self.transform:
            image = self.transform(image=img_rgb)['image']

        return image, self.labels[idx]



train_transform = A.Compose(
    [
        A.Resize(224, 224),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=55, p=0.5),
    ]
)

image_path = './dataset/maodie'

train_dataset = CatDataset(
    image_path=image_path,
    transform=train_transform,
    is_train=True
)

train_loader = DataLoader(dataset=train_dataset, batch_size=2)


for data, labels in train_loader:

    print(data.shape, labels)
    img_1 = data[0]
    img_2 = data[1]
    plt.subplot(121).imshow(img_1)
    plt.subplot(122).imshow(img_2)
    plt.show()


