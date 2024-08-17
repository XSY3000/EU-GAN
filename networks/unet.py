import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.network import base_module


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3, padding=1, stride=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, stride=stride)),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.utils.spectral_norm(
                nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.mpconv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False, skip_connection=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            if skip_connection:
                self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            else:
                self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)

        if x2 is not None:
            # Pad x1 to the same size as x2
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
            x1 = torch.cat([x2, x1], dim=1)
        else:
            pass
        # Concatenate along the channels axis
        x1 = self.conv(x1)

        return x1


class EAMUnet(base_module):
    def __init__(self):
        super().__init__()
        # Contracting path
        self.conv1 = DoubleConv(1, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        # self.middle = [CBAM(1024)]
        # self.middle += [ResnetBlock(1024)] * 9
        # self.middle = nn.Sequential(*self.middle)

        # Expansive path
        self.up1 = Up(1024, 512, skip_connection=False)
        self.up2 = Up(512, 256, skip_connection=False)
        self.up3 = Up(256, 128, skip_connection=False)
        self.up4 = Up(128, 64, skip_connection=False)
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(64, 1, kernel_size=1))

        self.spatial_attn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=5, dilation=5),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            CBAM(in_channel=256)
        )
        self.proj = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0)

        self.apply(self._init_weights)

    def forward(self, x):
        xEAM = self.spatial_attn(x)
        # Contracting path
        x = self.conv1(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        # x = self.middle(x)
        # Expansive path
        x = self.up1(x)
        x = self.up2(x)
        xEAM = F.interpolate(xEAM, size=x.size()[2:], mode='bilinear', align_corners=True)
        x = self.proj(torch.concat([x, xEAM], dim=1))
        x = self.up3(x)
        x = self.up4(x)
        out = self.conv2(x)
        out = torch.sigmoid(out)
        # copilot = self.conv3(x)
        # copilot = torch.sigmoid(copilot)
        return out


class UnetNoEAM(base_module):
    def __init__(self):
        super().__init__()
        # Contracting path
        self.conv1 = DoubleConv(1, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # Expansive path
        self.up1 = Up(1024, 512, skip_connection=False)
        self.up2 = Up(512, 256, skip_connection=False)
        self.up3 = Up(256, 128, skip_connection=False)
        self.up4 = Up(128, 64, skip_connection=False)
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(64, 1, kernel_size=1))

        self.apply(self._init_weights)

    def forward(self, x):
        # Contracting path
        x = self.conv1(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        # Expansive path
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        out = self.conv2(x)
        out = torch.sigmoid(out)
        return out


class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.conv_block(x)


class CBAM(nn.Module):
    def __init__(self, in_channel, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        # 通道注意力机制
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channel // reduction, out_channels=in_channel, kernel_size=1,  bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        # 空间注意力机制
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1,
                              padding=kernel_size // 2, bias=False)

    def forward(self, x):
        # 通道注意力机制
        maxout = self.mlp(self.max_pool(x))
        avgout = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(maxout + avgout)
        channel_out = channel_out * x
        # 空间注意力机制
        maxout, _ = torch.max(channel_out, dim=1, keepdim=True)
        avgout = torch.mean(channel_out, dim=1, keepdim=True)
        out = torch.cat((maxout, avgout), dim=1)
        out = self.sigmoid(self.conv(out))
        out = out * channel_out
        return out
