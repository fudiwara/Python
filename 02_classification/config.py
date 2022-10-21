import sys
sys.dont_write_bytecode = True
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import models

# 画像の一辺のサイズ (この大きさにリサイズされるので要確認)
cellSize = 200

# クラス全体の数
classesSize = 3

# 繰り返す回数
epochSize = 10

# ミニバッチのサイズ
batchSize = 80

# 学習時のサンプルを学習：検証データに分ける学習側の割合
splitRateTrain = 0.8

# データ変換
data_transforms = T.Compose([
    T.Resize(cellSize),
    T.RandomRotation(degrees = 15),
    T.RandomApply([T.GaussianBlur(5, sigma = (0.1, 5.0))], p = 0.5),
    T.ColorJitter(brightness = 0, contrast = 0, saturation = 0, hue = [-0.2, 0.2]),
    T.RandomHorizontalFlip(0.5),
    T.ToTensor()])

def calc_acc(output, label): # 結果が一致するラベルの数をカウントする
    p_arg = torch.argmax(output, dim = 1)
    return torch.sum(label == p_arg)

class build_model(nn.Module):
    def __init__(self):
        super(build_model, self).__init__()
        self.model_pret = models.resnet50(pretrained=True)
        self.bn = nn.BatchNorm1d(1000)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(1000, classesSize)

    def forward(self, input):
        mid_features = self.model_pret(input)
        x = self.bn(mid_features) # BNを追加
        x = self.dropout(x) # dropoutを追加
        output = self.classifier(x)
        return output

if __name__ == "__main__":
    mdl = build_model()
    print(mdl)

    from torchsummary import summary
    summary(mdl, (3, cellSize, cellSize))
