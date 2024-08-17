import torch
import torch.nn as nn
from torchvision import models


class base_module(nn.Module):
    def __init__(self):
        super().__init__()

    def _init_weights(self, model):
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                # 卷积层权重初始化为标准正态分布
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    # 偏置项初始化为零
                    nn.init.constant_(module.bias, 0.0)
            # elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.InstanceNorm2d):
            elif isinstance(module, nn.BatchNorm2d):
                # 批归一化层和实例归一化层权重初始化为标准正态分布，偏置项初始化为零
                nn.init.normal_(module.weight, mean=1.0, std=0.02)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                # 全连接层权重初始化为标准正态分布，偏置项初始化为零
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    # 冻结参数
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    # 解冻参数
    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    # load model
    def load(self, path, logger=None):
        if logger is None:
            logger = print
        if os.path.exists(path):
            weight_state_dict = torch.load(path)
            models_state_dict = self.state_dict()
            for name, param in weight_state_dict.items():
                if name in models_state_dict:
                    models_state_dict[name].copy_(param)
                else:
                    logger(f'Ignore layer: {name}')
            logger('Loaded model weights from {}'.format(path))
        else:
            logger('No such file: {}'.format(path))


import os.path
from networks.unet import EAMUnet, UnetNoEAM


class Generator_EAMUnet(EAMUnet):
    def __init__(self):
        super().__init__()

class Generator_UnetNoEAM(UnetNoEAM):
    def __init__(self):
        super().__init__()


class PatchGAN(base_module):
    def __init__(self, input_channels=1):
        super(PatchGAN, self).__init__()

        layers = []
        # First layer without batch normalization
        layers.append(nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Intermediate layers with batch normalization
        layers.append(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1))
        layers.append(nn.InstanceNorm2d(128))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1))
        layers.append(nn.InstanceNorm2d(256))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1))
        layers.append(nn.InstanceNorm2d(512))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Last layer with no batch normalization and sigmoid activation
        layers.append(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        output = self.net(x)
        output = output.view(x.shape[0], -1)
        return torch.mean(output, dim=1).view(x.shape[0], 1)



def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module



class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG19, self).__init__()
        vgg_pretrained_features = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X.expand(-1, 3, -1, -1))
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
