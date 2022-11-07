import sys, math
sys.dont_write_bytecode = True
import torch
import torchvision

# 繰り返す回数
epochSize = 20

# 学習時のバッチのサイズ
batchSize = 6

# カテゴリの総数 (背景の0を含めた合計)
numClasses = 2

# 検出の閾値
thDetection = 0.6

# データセットを学習用と評価用に分割する際の割合
splitRateTrain = 0.8

def build_model():
    model = torchvision.models.detection.retinanet_resnet50_fpn(weights="DEFAULT")

    in_features = model.head.classification_head.conv[0][0].in_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head.num_classes = numClasses
    
    cls_logits = torch.nn.Conv2d(in_features, num_anchors * numClasses, kernel_size = 3, stride=1, padding=1)
    torch.nn.init.normal_(cls_logits.weight, std=0.01)  # as per pytorch code
    torch.nn.init.constant_(cls_logits.bias, -math.log((1 - 0.01) / 0.01))  # as per pytorcch code 
    # assign cls head to model
    model.head.classification_head.cls_logits = cls_logits

    return model

if __name__ == "__main__":
    from torchsummary import summary
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = build_model()
    print(mdl)
    # summary(mdl, (3, imageSize, imageSize))
