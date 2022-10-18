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
val_rate_1 = 1

# 読み込み対象の画像拡張子
ext = [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"]

# 画像の一辺のサイズ (この大きさにリサイズされるので要確認)
cellSize = 200
# cellSize = 224
# cellSize = 384

# 繰り返す回数
# epochSize = 2
epochSize = 10
# epochSize = 100

# 学習するときの小さいセットの数：元のデータ数によるが早く終わらせたい場合は100以上とかもあり
# batchSize = 10
batchSize = 50
# batchSize = 40

# 学習時のサンプルを学習：検証データに分ける学習側の割合
splitRateTrain = 0.8

# データ変換
data_transforms = T.Compose([
    T.Resize(int(cellSize * 1.2)),
    T.RandomRotation(degrees = 15),
    T.RandomApply([T.GaussianBlur(5, sigma = (0.1, 5.0))], p = 0.5),
    T.ColorJitter(brightness = 0, contrast = 0, saturation = 0, hue = [-0.2, 0.2]),
    T.RandomHorizontalFlip(0.5),
    T.CenterCrop(cellSize),
    T.ToTensor()])

class build_model(nn.Module):
    def __init__(self):
        super().__init__()
        # self.model_pret = models.resnext101_32x8d(weights = models.ResNeXt101_32X8D_Weights.DEFAULT)
        self.model_pre = models.efficientnet_v2_s(weights = models.EfficientNet_V2_S_Weights.DEFAULT)
        self.bn = nn.BatchNorm1d(1000)
        self.dropout = nn.Dropout(0.5)
        self.classifier_0 = nn.Linear(1000, 1)
        self.classifier_1 = nn.Linear(1000, 1)
        self.relu = nn.ReLU()

    def forward(self, input):
        mid_features = self.model_pre(input)
        x = self.bn(mid_features) # BNを追加
        x = self.dropout(x) # dropoutを追加

        cf0 = self.classifier_0(x)
        out0 = self.relu(cf0)

        cf1 = self.classifier_1(x)
        out1 = self.relu(cf1)
        return out0, out1

if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = build_model().to(DEVICE)
    print(mdl)

    from torchsummary import summary
    summary(mdl, (3, cellSize, cellSize))
