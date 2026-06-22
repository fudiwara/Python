import sys
sys.dont_write_bytecode = True

import torch
import torch.nn as nn

# 画像の一辺サイズ
cellSize = 192

# 学習epoch
epochSize = 150

# ミニバッチ
batchSize = 16

# データセット数（load_dataset.py で設定）
dataset_size = 0


class ConvNormAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, norm="in", act="relu"):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p)]
        if norm == "bn":
            layers.append(nn.BatchNorm2d(out_ch))
        elif norm == "in":
            layers.append(nn.InstanceNorm2d(out_ch, affine=True))
        elif norm == "gn":
            layers.append(nn.GroupNorm(8 if out_ch >= 8 else 1, out_ch))

        if act == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif act == "lrelu":
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif act == "none":
            pass

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, ch, norm="in"):
        super().__init__()
        self.conv1 = ConvNormAct(ch, ch, k=3, s=1, p=1, norm=norm, act="relu")
        self.conv2 = ConvNormAct(ch, ch, k=3, s=1, p=1, norm=norm, act="none")
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + x
        return self.act(out)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm="in", n_res=1):
        super().__init__()
        self.down = ConvNormAct(in_ch, out_ch, k=3, s=2, p=1, norm=norm, act="relu")
        self.res = nn.Sequential(*[ResBlock(out_ch, norm=norm) for _ in range(n_res)])

    def forward(self, x):
        x = self.down(x)
        x = self.res(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, norm="in", n_res=1):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = ConvNormAct(in_ch + skip_ch, out_ch, k=3, s=1, p=1, norm=norm, act="relu")
        self.res = nn.Sequential(*[ResBlock(out_ch, norm=norm) for _ in range(n_res)])

    def forward(self, x, skip):
        x = self.up(x)
        # サイズずれの保険
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.res(x)
        return x


class GeneratorAE(nn.Module):
    """
    1ch(gray) -> 3ch(color), 出力は[-1, 1]
    """
    def __init__(self, base_ch=32, norm="in", n_res=1):
        super().__init__()
        c = base_ch

        self.stem = ConvNormAct(1, c, k=5, s=1, p=2, norm=norm, act="relu")
        self.e1 = nn.Sequential(ResBlock(c, norm=norm))
        self.e2 = DownBlock(c, c * 2, norm=norm, n_res=n_res)      # 96
        self.e3 = DownBlock(c * 2, c * 4, norm=norm, n_res=n_res)  # 48
        self.e4 = DownBlock(c * 4, c * 8, norm=norm, n_res=n_res)  # 24
        self.e5 = DownBlock(c * 8, c * 8, norm=norm, n_res=n_res)  # 12

        self.bottleneck = nn.Sequential(
            ResBlock(c * 8, norm=norm),
            ResBlock(c * 8, norm=norm),
        )

        self.d4 = UpBlock(c * 8, c * 8, c * 8, norm=norm, n_res=n_res)  # 24
        self.d3 = UpBlock(c * 8, c * 4, c * 4, norm=norm, n_res=n_res)  # 48
        self.d2 = UpBlock(c * 4, c * 2, c * 2, norm=norm, n_res=n_res)  # 96
        self.d1 = UpBlock(c * 2, c, c, norm=norm, n_res=n_res)          # 192

        self.out = nn.Sequential(
            nn.Conv2d(c + c, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x0 = self.stem(x)   # c,192
        x1 = self.e1(x0)    # c,192
        x2 = self.e2(x1)    # 2c,96
        x3 = self.e3(x2)    # 4c,48
        x4 = self.e4(x3)    # 8c,24
        x5 = self.e5(x4)    # 8c,12

        b = self.bottleneck(x5)

        y4 = self.d4(b, x4)
        y3 = self.d3(y4, x3)
        y2 = self.d2(y3, x2)
        y1 = self.d1(y2, x1)

        y = torch.cat([y1, x0], dim=1)
        y = self.out(y)
        return y


if __name__ == "__main__":
    from torchsummary import summary
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = GeneratorAE().to(device)
    print(m)
    summary(m, (1, cellSize, cellSize))