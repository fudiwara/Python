import sys
sys.dont_write_bytecode = True
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# 画像の一辺のサイズ (この大きさにリサイズされるので要確認)
cellSize = 192

# 繰り返す回数
epochSize = 200

# ミニバッチのサイズ
batchSize = 1

# データセットの数 (イテレーション数を求めたりするためにグローバルで使えるようにしておく)
dataset_size = 0

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = self.conv_bn_relu(1, 32, kernel_size=5) # 32x192x192
        self.enc2 = self.conv_bn_relu(32, 64, kernel_size=3, pool_kernel=4)  # 64x48x48
        self.enc3 = self.conv_bn_relu(64, 128, kernel_size=3, pool_kernel=2)  # 128x24x24
        self.enc4 = self.conv_bn_relu(128, 256, kernel_size=3, pool_kernel=2)  # 256x12x12
        self.enc5 = self.conv_bn_relu(256, 512, kernel_size=3, pool_kernel=2)  # 512x6x6
        self.dec1 = self.conv_bn_relu(512, 256, kernel_size=3, pool_kernel=-2)  # 256x12x12
        self.dec2 = self.conv_bn_relu(256 + 256, 128, kernel_size=3, pool_kernel=-2)  # 128x24x24
        self.dec3 = self.conv_bn_relu(128 + 128, 64, kernel_size=3, pool_kernel=-2)  # 64x48x48
        self.dec4 = self.conv_bn_relu(64 + 64, 32, kernel_size=3, pool_kernel=-4)  # 32x192x192
        self.dec5 = nn.Sequential(nn.Conv2d(32 + 32, 3, kernel_size=5, padding=2), nn.Tanh())

    def conv_bn_relu(self, in_ch, out_ch, kernel_size=3, pool_kernel=None):
        layers = []
        if pool_kernel is not None:
            if pool_kernel > 0: layers.append(nn.AvgPool2d(pool_kernel))
            elif pool_kernel < 0: layers.append(nn.UpsamplingNearest2d(scale_factor=-pool_kernel))
        layers.append(nn.Conv2d(in_ch, out_ch, kernel_size, padding=(kernel_size - 1) // 2))
        layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        out = self.dec1(x5)
        out = self.dec2(torch.cat([out, x4], dim=1))
        out = self.dec3(torch.cat([out, x3], dim=1))
        out = self.dec4(torch.cat([out, x2], dim=1))
        out = self.dec5(torch.cat([out, x1], dim=1))
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = self.conv_bn_relu(4, 16, kernel_size=5, reps=1) # fake/true color + gray
        self.conv2 = self.conv_bn_relu(16, 32, pool_kernel=4)
        self.conv3 = self.conv_bn_relu(32, 64, pool_kernel=2)
        self.conv4 = self.conv_bn_relu(64, 128, pool_kernel=2)
        self.conv5 = self.conv_bn_relu(128, 256, pool_kernel=2)
        self.conv6 = self.conv_bn_relu(256, 512, pool_kernel=2)
        self.out_patch = nn.Conv2d(512, 1, kernel_size=1) #1x3x3

    def conv_bn_relu(self, in_ch, out_ch, kernel_size=3, pool_kernel=None, reps=2):
        layers = []
        for i in range(reps):
            if i == 0 and pool_kernel is not None:
                layers.append(nn.AvgPool2d(pool_kernel))
            layers.append(nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, kernel_size, padding=(kernel_size - 1) // 2))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv6(self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x))))))
        return self.out_patch(out)


if __name__ == "__main__":
    from torchsummary import summary
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    mdl_gen = Generator().to(DEVICE)
    print(mdl_gen)
    summary(mdl_gen, (1, cellSize, cellSize))

    mdl_dis = Discriminator().to(DEVICE)
    print(mdl_dis)
    summary(mdl_dis, (4, cellSize, cellSize))