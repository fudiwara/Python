import sys, math
sys.dont_write_bytecode = True
import torch
import torchvision

# 繰り返す回数
epochSize = 20

# 学習時のバッチのサイズ
batchSize = 4

# カテゴリの総数 (背景の0を含めた合計)
numClasses = 2

# 検出の閾値
thDetection = 0.6

# データセットを学習用と評価用に分割する際の割合
splitRateTrain = 0.8

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
    torch.nn.init.normal_(cls_logits.weight, std=0.01)
    torch.nn.init.constant_(cls_logits.bias, -math.log((1 - 0.01) / 0.01))
    model.head.classification_head.cls_logits = cls_logits

    return model

if __name__ == "__main__":
    from torchsummary import summary
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = create_model()
    print(mdl)
    # summary(mdl, (3, imageSize, imageSize))
