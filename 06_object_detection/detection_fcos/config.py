import sys, math
sys.dont_write_bytecode = True
import torch
import torchvision
from torchvision.transforms import v2 as T

# 繰り返す回数
epochSize = 20

# 学習時のバッチのサイズ
batchSize = 4

# 勾配累積ステップ数
accumulationSteps = 3

# カテゴリの総数 (背景の0を含めた合計)
numClasses = 2

# 検出の閾値
thDetection = 0.6

# データセットを学習用と評価用に分割する際の割合
splitRateTrain = 0.8

transforms_train = T.Compose([
    T.ToImage(), # PIL、Tensor、NumPy のいずれの入力も可
    T.RandomHorizontalFlip(0.5),
    T.RandomRotation(degrees = 5), # box は回転後の外接矩形になる
    T.RandomApply([T.GaussianBlur(5, sigma = (0.1, 2.0))], p = 0.3),
    T.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.1, hue = 0),
    T.ToDtype(torch.float32, scale = True), # 画像のみ変換される (maskはuint8のまま)
    T.SanitizeBoundingBoxes(), # 潰れた box と対応する label / mask を除去
])

transforms_eval = T.Compose([
    T.ToImage(), # PIL、Tensor、NumPy のいずれの入力も可
    T.ToDtype(torch.float32, scale = True),
])

def build_model(sw_train_eval):
    if sw_train_eval == "train":
        model = torchvision.models.detection.fcos_resnet50_fpn(weights="DEFAULT")
    else:
        model = torchvision.models.detection.fcos_resnet50_fpn()

    # 事前訓練済みのヘッドを新しいものと置き換える
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head.num_classes = numClasses
    out_channels = model.head.classification_head.conv[9].out_channels
    cls_logits = torch.nn.Conv2d(out_channels, num_anchors * numClasses, kernel_size=3, stride=1, padding=1)
    torch.nn.init.normal_(cls_logits.weight, std = 0.01)
    torch.nn.init.constant_(cls_logits.bias, -math.log((1 - 0.01) / 0.01))
    model.head.classification_head.cls_logits = cls_logits

    return model

if __name__ == "__main__":
    from torchsummary import summary
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = create_model()
    print(mdl)
    # summary(mdl, (3, imageSize, imageSize))
