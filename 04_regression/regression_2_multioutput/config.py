import sys
sys.dont_write_bytecode = True
import torch
import torch.nn as nn
from torchvision.transforms import v2 as T
import timm

# ファイル名の _ で区切られる数がいくつあるか
sep_num = 4

# ファイル名を _ で分割したN番目を教師データとして使うか (前から0・後ろから-1)
sep_val_0 = 0
sep_val_1 = 1

# 学習時の値に正規化するための係数
val_rate_0 = 150

# 読み込み対象の画像拡張子
ext = [".jpg", ".jpeg", ".png", ".bmp"]

# 画像の一辺のサイズ (この大きさにリサイズされるので要確認)
cellSize = 224

# 繰り返す回数
epochSize = 15

# 学習するときの小さいセットの数：元のデータ数によるが早く終わらせたい場合は100以上とかもあり
batchSize = 64

# 学習時のサンプルを学習：検証データに分ける学習側の割合
splitRateTrain = 0.8

# データ変換
transforms_train = T.Compose([
    T.ToImage(), # テンソル変換の前にPIL画像に変換
    T.Resize(int(cellSize * 1.2), antialias = True), # 画像を少し大きくリサイズしてからランダムクロップ
    T.RandomRotation(degrees = 5), # 画像をランダムに回転
    T.RandomApply([T.GaussianBlur(5, sigma = (0.1, 5.0))], p = 0.5), # 画像をランダムにぼかす
    T.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0, hue = 0), # 画像の明るさとコントラストをランダムに変化
    T.RandomHorizontalFlip(0.5), # 画像をランダムに左右反転
    T.RandomCrop(cellSize), # 画像をランダムに切り抜き
    T.ToDtype(torch.float32, scale = True), # float32の[0.0, 1.0]にスケール変換
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # 画像を正規化
    T.RandomErasing(p = 0.5, scale = (0.02, 0.33), ratio = (0.3, 3.3)) # 画像の一部をランダムに消す
])

transforms_eval = T.Compose([
    T.ToImage(),
    T.Resize(cellSize, antialias = True), # 短辺をcellSizeにリサイズ
    T.CenterCrop(cellSize), # 中心を切り抜き
    T.ToDtype(torch.float32, scale = True), # float32の[0.0, 1.0]にスケール変換
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class build_model(nn.Module):
    def __init__(self, sw_train_eval):
        super().__init__()
        pretrained = (sw_train_eval == "train")
        
        self.body = timm.create_model(
            "efficientnet_b0",
            pretrained = pretrained,
            num_classes = 0 # 最後の全結合層を削除して特徴両ベクトルを出力
        )
        in_features = self.body.num_features # モデルの出力の次元数

        self.bn = nn.BatchNorm1d(in_features)
        self.dropout = nn.Dropout(0.5)
        
        self.head_age = nn.Linear(in_features, 1) # 一つの出力となる年齢予測用の回帰ヘッド
        self.head_gender = nn.Linear(in_features, 2) # 2つの出力にした性別予測用の分類ヘッド

    def forward(self, input):
        features = self.body(input)
        features = self.bn(features)
        features = self.dropout(features)
        out_age = self.head_age(features).squeeze(1) # 出力特徴量から年齢予測 [batchSize]
        out_gender = self.head_gender(features) # 出力特徴量から性別予測 [batchSize, 2]
        return out_age, out_gender

if __name__ == "__main__":
    import os
    from torchinfo import summary
    from torchviz import make_dot

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(DEVICE)

    mdl = build_model("train")
    print(mdl)
    summary(mdl, (batchSize, 3, cellSize, cellSize))
    
    x = torch.randn(batchSize, 3, cellSize, cellSize).to(DEVICE) # 適当な入力
    y = mdl(x) # その出力
    
    img = make_dot(y, params = dict(mdl.named_parameters())) # 計算グラフの表示
    img.format = "png"
    img.render("_model_graph") # グラフを画像に保存
    os.remove("_model_graph") # 拡張子無しのファイルもできるので個別に削除
