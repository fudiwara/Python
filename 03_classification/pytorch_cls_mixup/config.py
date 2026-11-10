import sys
sys.dont_write_bytecode = True
import torch
from torchvision.transforms import v2 as T
import timm

# 画像の一辺のサイズ (この大きさにリサイズされるので要確認)
cellSize = 128

# クラス全体の数
classesSize = 10

# 繰り返す回数
epochSize = 250

# 学習するときの小さいセットの数
batchSize = 64

# 学習時のサンプルを学習：検証データに分ける学習側の割合
splitRateTrain = 0.9

transforms_train = T.Compose([ # 学習時のデータ変換
    T.ToImage(), # 画像データをPyTrochテンソル形式に変換
    T.Resize(int(cellSize * 1.15), antialias=True), # 画像を少し大きくリサイズしてからランダムクロップ
    T.RandomHorizontalFlip(0.5), # 画像をランダムに左右反転
    T.RandAugment(num_ops = 2, magnitude = 7),
    T.ToDtype(torch.float32, scale=True), # float32の[0.0, 1.0]にスケール変換
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # 画像を正規化
    T.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3)) # 画像の一部をランダムに消す
])

transforms_eval = T.Compose([ # 推定時のデータ変換
    T.ToImage(), # 画像データをPyTrochテンソル形式に変換
    T.Resize(cellSize, antialias=True), # 短辺をcellSizeにリサイズ
    T.CenterCrop(cellSize), # 中心を切り抜き
    T.ToDtype(torch.float32, scale=True), # float32の[0.0, 1.0]にスケール変換
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 画像を正規化
])

mixup_fn = T.MixUp(num_classes = classesSize, alpha = 0.2)

def calc_acc(output, label): # 結果が一致するラベルの数をカウントする
    p_arg = torch.argmax(output, dim = 1)
    return torch.sum(label == p_arg).item()

class BuildModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            # "convnext_nano",
            "mobilenetv4_conv_medium.e500_r256_in1k",
            pretrained = False,
            num_classes = classesSize,
            drop_path_rate = 0.35, # スクラッチ学習時の過学習を抑える Stochastic Depth
            drop_rate = 0.2 # Head直前のDropoutを追加
        )

    def forward(self, input):
        x = self.model(input)
        return x

if __name__ == "__main__":
    from torchinfo import summary
    from torchview import draw_graph

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    mdl = BuildModel().to(DEVICE)
    print(mdl)
    summary(mdl, (batchSize, 3, cellSize, cellSize))
    
    model_graph = draw_graph(mdl, input_size=(batchSize, 3, cellSize, cellSize), expand_nested = True, save_graph = False)
    model_graph.visual_graph.render("model_flow", format = "png", cleanup = True)
