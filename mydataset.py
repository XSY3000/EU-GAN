import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from skimage.morphology import skeletonize
from torch.utils.data import Dataset
from torchvision.transforms import *
import matplotlib.pyplot as plt


class CustomDataset_G(Dataset):
    def __init__(self, image_dir, mode='train', mask_density=50):
        self.mask_density = mask_density
        self.mode = mode
        
        if mode in ['infer']:
            image_dirs = image_dir if isinstance(image_dir, list) else [image_dir]
        else:
            if isinstance(image_dir, str):
                image_dirs = [os.path.join(image_dir, mode, 'labels')]
            else:
                image_dirs = [os.path.join(d, mode, 'labels') for d in image_dir]
        self.label_name_list = []
        for d in image_dirs:
            if 'cotton' in d and mode == 'train':
                self.label_name_list += [os.path.join(d, name) for name in os.listdir(d) if '.png' in name] * 5
            else:
                self.label_name_list += [os.path.join(d, name) for name in os.listdir(d) if '.png' in name]
        pass

    def __len__(self):
        return len(self.label_name_list)


    def __getitem__(self, idx):
        label_path = self.label_name_list[idx]
        label_org = np.array(Image.open(label_path).convert('L'))
        haveinput = os.path.exists(image_path := label_path.replace('labels', 'images'))
        # 白色像素点的bounding box, 裁图
        arr = np.where(label_org == 255)
        x_min, x_max = np.min(arr[1]), np.max(arr[1])
        y_min, y_max = np.min(arr[0]), np.max(arr[0])
        label = label_org[y_min:y_max, x_min:x_max]
        label = cv2.resize(label, (384, 640))
        if self.mode == 'infer':
            label = label / 255
            label = np.expand_dims(label, axis=0)
            label = np.expand_dims(label, axis=0)
            label = torch.cuda.FloatTensor(label)
            return label, label_path, label_org // 255, (x_min, x_max, y_min, y_max)
        if haveinput:
            image_org = np.array(Image.open(image_path).convert('L'))
            image = image_org[y_min:y_max, x_min:x_max]
            image = cv2.resize(image, (384, 640))
            image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]
            if self.mode == 'test':
                mask = label_org - image_org
            else:
                mask = label - image
        else:
            mask = self.get_protection(label)
            image = (1 - mask) * label
            mask = mask * label

        # 转换为tensor
        image = image / 255
        image = np.expand_dims(image, axis=0)
        image = torch.cuda.FloatTensor(image)

        if self.mode == 'test':
            image = torch.unsqueeze(image, dim=0)
            return image, label_org, mask, label_path, image_org // 255, (x_min, x_max, y_min, y_max)

        label = label / 255
        label = np.expand_dims(label, axis=0)
        label = torch.cuda.FloatTensor(label)
        mask = mask / 255
        mask = np.expand_dims(mask, axis=0)
        mask = torch.cuda.FloatTensor(mask)


        return image, label, mask


    def get_mask(self, label):
        # H, W = label.shape
        # l = max(min(H, W) // 20, 6)
        l = 20
        # 生成mask
        mask = np.zeros_like(label, np.uint8)
        arr = list(np.argwhere(label == 255))
        random.shuffle(arr)
        N = random.randint(10, max(11, len(arr) // self.mask_density))
        for i in range(N):
            x, y = arr[i]
            shape = random.choice(['r', 'c', 'e'])
            if shape == 'r':
                mask = cv2.rectangle(mask, (y - random.randint(5, l), x - random.randint(5, l)),
                                     (y + random.randint(5, l), x + random.randint(5, l)), 255, -1)
            elif shape == 'c':
                mask = cv2.circle(mask, (y, x), random.randint(5, l), 255, -1)
            elif shape == 'e':
                mask = cv2.ellipse(mask, (y, x), (random.randint(5, l), random.randint(5, l)),
                                   random.randint(0, 360), 0, 360, 255, -1)

        return mask

    def get_protection(self, label):
        protection = np.zeros_like(label)
        label_ = cv2.morphologyEx(label, cv2.MORPH_CLOSE, np.ones((3, 3), int), iterations=3)
        skeleton = skeletonize(label_ / 255)
        skeleton[skeleton > 0] = 1
        arr = np.argwhere(skeleton == 1)
        for point in arr:
            x, y = point
            # 8邻域内是否有其他点
            if np.sum(skeleton[x - 1:x + 2, y - 1:y + 2]) == 2:
                cv2.circle(protection, (y, x), 10, 1, -1)
        mask = self.get_mask(skeleton * 255) // 255
        mask[protection == 1] = 0
        return mask
