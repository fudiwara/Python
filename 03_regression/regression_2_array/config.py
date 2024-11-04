import sys
sys.dont_write_bytecode = True
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchvision import models
import numpy as np

# ファイル名の _ で区切られる数がいくつあるか
sep_num = 4

# ファイル名を _ で分割したN番目を教師データとして使うか (前から0・後ろから-1)
sep_val_0 = 0
sep_val_1 = 1

# 学習時の値に正規化するための係数
val_rate_0 = 150
val_rate_1 = 2

# 読み込み対象の画像拡張子
ext = [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"]

# 画像の一辺のサイズ (この大きさにリサイズされるので要確認)
# cellSize = 200
cellSize = 224

# 繰り返す回数
epochSize = 15

# 学習するときの小さいセットの数：元のデータ数によるが早く終わらせたい場合は100以上とかもあり
batchSize = 50

# 学習時のサンプルを学習：検証データに分ける学習側の割合
splitRateTrain = 0.8

# データ変換
data_transforms = T.Compose([
    T.Resize(int(cellSize)),
    # T.Resize(int(cellSize * 1.2)),
    # T.RandomRotation(degrees = 15),
    # T.RandomApply([T.GaussianBlur(5, sigma = (0.1, 5.0))], p = 0.5),
    # T.ColorJitter(brightness = 0, contrast = 0, saturation = 0, hue = [-0.2, 0.2]),
    # T.RandomHorizontalFlip(0.5),
    T.CenterCrop(cellSize),
    T.ToTensor()])

class build_model(nn.Module):
    def __init__(self, sw_train_eval):
        super().__init__()
        if sw_train_eval == "train":
            self.model_pre = models.efficientnet_v2_s(weights = "DEFAULT")
        else:
            self.model_pre = models.efficientnet_v2_s()
        self.model_pre.classifier[1] = nn.Linear(1280, 2, bias = True)

    def forward(self, input):
        x = self.model_pre(input)
        return x

if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = build_model().to(DEVICE)
    print(mdl)

    from torchsummary import summary
    summary(mdl, (3, cellSize, cellSize))
