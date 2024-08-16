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
        # if mode == 'test':
        #     image_dir = [dirs for dirs in image_dir if 'cotton' in dirs]
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
            # self.label_name_list += [os.path.join(d, name) for d in image_dirs for name in os.listdir(d)]
        pass

    def __len__(self):
        return len(self.label_name_list)


    def __getitem__(self, idx):
        label_path = self.label_name_list[idx]
        # image_name = os.path.basename(label_path)
        label_org = np.array(Image.open(label_path).convert('L'))
        if 'orange' in label_path:
            label_org = cv2.resize(label_org, (2024, 3400))
            label_org = cv2.threshold(label_org, 127, 255, cv2.THRESH_BINARY)[1]
        haveinput = os.path.exists(image_path := label_path.replace('labels', 'images'))
        # 白色像素点的bounding box, 裁图
        arr = np.where(label_org == 255)
        x_min, x_max = np.min(arr[1]), np.max(arr[1])
        y_min, y_max = np.min(arr[0]), np.max(arr[0])
        if self.mode != 'test':
            label = label_org[y_min:y_max, x_min:x_max]
            label = cv2.resize(label, (384, 640))
        else:
            # resize至16的倍数
            label = label_org[100:-100, 100:-100]
        # ph, pw = label.shape[0] % 16, label.shape[1] % 16
        # label = label[:-ph, :-pw]
        if self.mode == 'infer':
            label = label / 255
            label = np.expand_dims(label, axis=0)
            label = np.expand_dims(label, axis=0)
            label = torch.cuda.FloatTensor(label)
            return label, label_path, label_org // 255, (x_min, x_max, y_min, y_max)
        if haveinput:
            image_org = np.array(Image.open(image_path).convert('L'))
            if 'orange' in label_path:
                image_org = cv2.resize(image_org, (2024, 3400))
                image_org = cv2.threshold(image_org, 127, 255, cv2.THRESH_BINARY)[1]
            if self.mode != 'test':
            # if self.mode:
                image = image_org[y_min:y_max, x_min:x_max]
                image = cv2.resize(image, (384, 640))
                image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]
            else:
                image = image_org[100:-100, 100:-100]
            if self.mode == 'test':
                mask = label_org - image_org
            else:
                mask = label - image
        else:
            # mask = self.get_mask(label)
            mask = self.get_protection(label)
            image = (1 - mask) * label
            mask = mask * label
            # plt.figure()
            # plt.subplot(1, 3, 1)
            # plt.imshow(image, 'gray')
            # plt.title('image')
            # plt.subplot(1, 3, 2)
            # plt.imshow(label, 'gray')
            # plt.title('label')
            # plt.subplot(1, 3, 3)
            # plt.imshow(mask, 'gray')
            # plt.title('mask')
            # plt.show()

        # 转换为tensor
        # image = ToTensor()(image)
        # label = ToTensor()(label)
        image = image / 255
        image = np.expand_dims(image, axis=0)
        image = torch.cuda.FloatTensor(image)

        if self.mode == 'test':
            image = torch.unsqueeze(image, dim=0)
            # plt.imshow(mask, 'gray')
            # plt.show()
            # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            # for i in range(1, num_labels):
            #     if stats[i][-1] > 1500:
            #         mask[labels == i] = 0
            # return image_org, label_path, image, label_org, mask, (x_min, x_max, y_min, y_max)
            # plt.imshow(mask, 'gray')
            # plt.show()
            return image, label_org, mask, label_path, image_org // 255, (x_min, x_max, y_min, y_max)

        label = label / 255
        label = np.expand_dims(label, axis=0)
        label = torch.cuda.FloatTensor(label)
        mask = mask / 255
        mask = np.expand_dims(mask, axis=0)
        mask = torch.cuda.FloatTensor(mask)

        # rebuild_dataloader = np.zeros(1)
        # if self.rm_water and random.randint(0, 2) == 0 and not os.path.exists(image_path):
        #     self.label_name_list.pop(idx)
        #     rebuild_dataloader = np.ones(1)

        return image, label, mask



    # def __getitem__(self, idx):
    #     label_path = self.label_name_list[idx]
    #     image_org = np.array(Image.open(label_path.replace('labels', 'raws')))
    #     # seg_org = np.array(Image.open(label_path.replace('labels', 'images')).convert('L'))
    #     label_org = np.array(Image.open(label_path).convert('L'))
    #     # 白色像素点的bounding box, 裁图
    #     # arr = np.where(label_org == 255)
    #     # x_min, x_max = np.min(arr[1]), np.max(arr[1])
    #     # y_min, y_max = np.min(arr[0]), np.max(arr[0])
    #     # label = label_org[y_min:y_max, x_min:x_max]
    #     image = image_org[4:-4:4, 20:-20:4]
    #     # seg = seg_org[4:-4:4, 20:-20:4]
    #     label = label_org[4:-4:4, 20:-20:4]
    #     # label[seg == 255] = 0
    #
    #
    #     transforms = Compose([
    #         ToTensor(),
    #         Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    #     ])
    #     image = transforms(image)
    #     label = ToTensor()(label)
    #     return image, label, label_path  # , mask

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
        # plt.figure()
        # plt.subplot(1, 2, 1)
        # plt.imshow(protection, 'gray')
        # plt.subplot(1, 2, 2)
        # plt.imshow(label, 'gray')
        # plt.show()
        return mask


if __name__ == '__main__':
    # dataset_path = ['filedata/cotton0610', 'filedata/cotton0618']
    dataset_path = [r'F:\file\myRice\RootSeg-paddle\data\allinone\preds\small\cotton']
    dataset = CustomDataset_G(dataset_path, 'infer')
    for i in range(10):
        image, label, mask = dataset[i]
