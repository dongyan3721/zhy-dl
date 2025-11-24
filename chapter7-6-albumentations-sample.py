#%%
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


def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 18
    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))
        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))
        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)
        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)
        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)
        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)
    plt.show()
#%%

path_img = './dataset/yu/ch7/00016.png'
path_mask = './dataset/yu/ch7/00016_matte.png'

image = cv2.imread(path_img)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mask = cv2.imread(path_mask, 0)

data_transform = A.Compose(
    [
        A.RandomRotate90(p=1),
        A.HueSaturationValue(p=0.5)
    ]
)

# 这一行模拟dataset中的 self.transform()
data_augmented = data_transform(image=image_rgb, mask=mask)

# 这一行模拟迭代训练中从DataLoader获取到的data
image_padded, mask_padded = data_augmented['image'], data_augmented['mask']

# 可视化
visualize(image_padded, mask_padded, original_image=image_rgb, original_mask=mask)

#%%
from importlib import import_module

path_img = './dataset/yu/ch7/kun.png'

image = cv2.imread(path_img)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mask = cv2.imread(path_mask, 0)


A = import_module("albumentations")  # 虽复杂，但是是新知识点，保留作为学习资料

aug_list_cand = dir(A)
aug_list_filtered = [aug for aug in aug_list_cand if not aug.startswith("IAA")]
counter = 0

f, ax = plt.subplots(10, 10, figsize=(24, 24))
f.subplots_adjust(wspace=0.2,hspace=0.5)
for idx, aug_name in enumerate(aug_list_cand):
    try:
        data_transform = eval("A.{}".format(aug_name))(p=1)
        data_augmented = data_transform(image=image_rgb)
        image_aug = data_augmented['image']
        counter += 1
        print(aug_name)
        # PiecewiseAffine此增强方法非常慢
        ax.ravel()[counter-1].imshow(image_aug.astype(np.uint8))
        ax.ravel()[counter-1].set_title(aug_name)
    except Exception as e:
        pass

# plt.savefig("aug_demo-68.png")
plt.show()
#%%
