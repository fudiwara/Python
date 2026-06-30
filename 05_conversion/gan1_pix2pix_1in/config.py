import sys
sys.dont_write_bytecode = True

import torch
import torch.nn as nn

# 学習用の設定
cellSize = 192
epochSize = 60
batchSize = 1
dataset_size = 1000

# モデルの設定
in_channels = 1
out_channels = 3
base_channels = 32
num_res_blocks = 2
norm_type = "instance" # "batch" or "instance"
use_amp = True

# 最適化関数の設定
lr_g = 2e-4
lr_d = 8e-5
beta1 = 0.5
beta2 = 0.999

# 損失関数の設定
lambda_l1 = 50.0 # 30〜100とか: 低いほど色はのりやすくなるがノイズとかも増える
d_update_interval = 4 # 識別器の更新間隔(1〜10とか): 大きいほど学習は安定するが色ののりが悪くなる
real_label_smooth = 0.9 # 学習安定化のため識別器の正解ラベルを少し下げる(0.7〜0.9とか)

# 学習安定化 (データセットの数が少ない場合は有効)
use_input_noise_for_d = True # 学習安定化のため識別器の入力にノイズを加える
input_noise_std = 0.01

def get_norm_layer(ch: int, norm: str):
    if norm == "batch":
        return nn.BatchNorm2d(ch)
    elif norm == "instance":
        return nn.InstanceNorm2d(ch, affine=True)
    else:
        raise ValueError(f"Unsupported norm type: {norm}")

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, norm="instance", act="relu"):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)]
        layers.append(get_norm_layer(out_ch, norm))
        if act == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif act == "lrelu":
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        else:
            raise ValueError(f"Unsupported act: {act}")
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class ResBlock(nn.Module):
    def __init__(self, ch, norm="instance"):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = get_norm_layer(ch, norm)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = get_norm_layer(ch, norm)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.act(out + identity)
        return out

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm="instance"):
        super().__init__()
        self.block = ConvBlock(in_ch, out_ch, k=4, s=2, p=1, norm=norm, act="relu")

    def forward(self, x):
        return self.block(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm="instance"):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = ConvBlock(in_ch, out_ch, k=3, s=1, p=1, norm=norm, act="relu")

    def forward(self, x):
        return self.conv(self.up(x))

class Generator(nn.Module):
    def __init__(self, in_ch=in_channels, out_ch=out_channels, base_ch=base_channels,
                 n_res=num_res_blocks, norm=norm_type):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base_ch, k=5, s=1, p=2, norm=norm, act="relu")
        self.enc2 = DownBlock(base_ch, base_ch * 2, norm=norm)
        self.enc3 = DownBlock(base_ch * 2, base_ch * 4, norm=norm)
        self.enc4 = DownBlock(base_ch * 4, base_ch * 8, norm=norm)
        self.enc5 = DownBlock(base_ch * 8, base_ch * 8, norm=norm)

        self.resblocks = nn.Sequential(*[ResBlock(base_ch * 8, norm=norm) for _ in range(n_res)])

        self.dec1 = UpBlock(base_ch * 8, base_ch * 8, norm=norm)
        self.dec2 = UpBlock(base_ch * 16, base_ch * 4, norm=norm)
        self.dec3 = UpBlock(base_ch * 8, base_ch * 2, norm=norm)
        self.dec4 = UpBlock(base_ch * 4, base_ch, norm=norm)
        self.dec5 = nn.Sequential(
            nn.Conv2d(base_ch * 2, out_ch, kernel_size=5, stride=1, padding=2),
            nn.Tanh()
        )

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)

        b = self.resblocks(x5)

        d1 = self.dec1(b)
        d2 = self.dec2(torch.cat([d1, x4], dim=1))
        d3 = self.dec3(torch.cat([d2, x3], dim=1))
        d4 = self.dec4(torch.cat([d3, x2], dim=1))
        out = self.dec5(torch.cat([d4, x1], dim=1))
        return out


class Discriminator(nn.Module):
    """PatchGAN discriminator"""
    def __init__(self, in_ch=4, base_ch=64, norm=norm_type):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_ch, base_ch * 2, kernel_size=4, stride=2, padding=1, bias=False),
            get_norm_layer(base_ch * 2, norm),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_ch * 2, base_ch * 4, kernel_size=4, stride=2, padding=1, bias=False),
            get_norm_layer(base_ch * 4, norm),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_ch * 4, base_ch * 8, kernel_size=4, stride=1, padding=1, bias=False),
            get_norm_layer(base_ch * 8, norm),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_ch * 8, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.net(x)