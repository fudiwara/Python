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
classesSize = 10

# 繰り返す回数
epochSize = 50

# ミニバッチのサイズ
batchSize = 100

# 学習時のサンプルを学習：検証データに分ける学習側の割合
splitRateTrain = 0.8

# データ変換
data_transforms = T.Compose([
    T.Resize(int(cellSize * 1.2)),
    T.RandomRotation(degrees = 5),
    T.RandomApply([T.GaussianBlur(5, sigma = (0.1, 5.0))], p = 0.5),
    T.ColorJitter(brightness = 0, contrast = 0, saturation = 0, hue = [-0.2, 0.2]),
    T.RandomHorizontalFlip(0.5),
    T.CenterCrop(cellSize),
    T.ToTensor(),
    T.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))
    ])

def calc_acc(output, label): # 結果が一致するラベルの数をカウントする
    p_arg = torch.argmax(output, dim = 1)
    return torch.sum(label == p_arg)

class build_model(nn.Module):
    def __init__(self, sw_train_eval):
        super().__init__()
        if sw_train_eval == "train":
            self.model_pre = models.efficientnet_v2_s(weights = "DEFAULT")
        else:
            self.model_pre = models.efficientnet_v2_s()
        self.model_pre.classifier[1] = nn.Linear(1280, classesSize, bias = True)

    def forward(self, input):
        x = self.model_pre(input)
        return x

if __name__ == "__main__":
    import os
    from torchinfo import summary
    from torchviz import make_dot

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(DEVICE)

    mdl = build_model()
    print(mdl)
    summary(mdl, (batchSize, 3, cellSize, cellSize))
    
    x = torch.randn(batchSize, 3, cellSize, cellSize).to(DEVICE) # 適当な入力
    y = mdl(x) # その出力
    
    img = make_dot(y, params = dict(mdl.named_parameters())) # 計算グラフの表示
    img.format = "png"
    img.render("_model_graph") # グラフを画像に保存
    os.remove("_model_graph") # 拡張子無しのファイルもできるので個別に削除

