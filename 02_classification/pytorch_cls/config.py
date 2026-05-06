import sys
sys.dont_write_bytecode = True
import torch
import torch.nn as nn
from torchvision.transforms import v2 as T
import timm

# 画像の一辺のサイズ (この大きさにリサイズされるので要確認)
cellSize = 224

# クラス全体の数
classesSize = 10

# 繰り返す回数
epochSize = 20

# ミニバッチのサイズ
batchSize = 64

# 学習時のサンプルを学習：検証データに分ける学習側の割合
splitRateTrain = 0.8

# データ変換
transforms_train = T.Compose([
    T.ToImage(), # テンソル変換の前にPIL画像に変換
    T.Resize(int(cellSize * 1.2), antialias=True), # 画像を少し大きくリサイズしてからランダムクロップ
    T.RandomRotation(degrees = 5), # 画像をランダムに回転
    T.RandomApply([T.GaussianBlur(5, sigma = (0.1, 5.0))], p = 0.5), # 画像をランダムにぼかす
    T.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0, hue = 0), # 画像の明るさとコントラストをランダムに変化
    T.RandomHorizontalFlip(0.5), # 画像をランダムに左右反転
    T.RandomCrop(cellSize), # 画像をランダムに切り抜き
    T.ToDtype(torch.float32, scale=True), # float32の[0.0, 1.0]にスケール変換
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # 画像を正規化
    T.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)) # 画像の一部をランダムに消す
])

transforms_eval = T.Compose([
    T.ToImage(),
    T.Resize(cellSize, antialias=True), # 短辺をcellSizeにリサイズ
    T.CenterCrop(cellSize), # 中心を切り抜き
    T.ToDtype(torch.float32, scale=True), # float32の[0.0, 1.0]にスケール変換
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def calc_acc(output, label): # 結果が一致するラベルの数をカウントする
    p_arg = torch.argmax(output, dim = 1)
    return torch.sum(label == p_arg)

class build_model(nn.Module):
    def __init__(self, sw_train_eval):
        super().__init__()
        pretrained = (sw_train_eval == "train")
        self.model = timm.create_model(
            "convnext_tiny.fb_in22k_ft_in1k", 
            pretrained = pretrained,
            num_classes = classesSize
        )

    def forward(self, input):
        x = self.model(input)
        return x

if __name__ == "__main__":
    import os
    from torchinfo import summary
    from torchviz import make_dot

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(DEVICE)

    mdl = build_model("train").to(DEVICE)
    print(mdl)
    summary(mdl, (batchSize, 3, cellSize, cellSize))
    
    x = torch.randn(batchSize, 3, cellSize, cellSize).to(DEVICE) # 適当な入力
    y = mdl(x) # その出力
    
    img = make_dot(y, params = dict(mdl.named_parameters())) # 計算グラフの表示
    img.format = "png"
    img.render("_model_graph") # グラフを画像に保存
    os.remove("_model_graph") # 拡張子無しのファイルもできるので個別に削除
