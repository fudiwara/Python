import sys
sys.dont_write_bytecode = True
import torch
import torchvision

# 繰り返す回数
epochSize = 10

# 学習時のバッチのサイズ
batchSize = 4

# カテゴリの総数 (背景の0を含めた合計)
numClasses = 14

# 検出の閾値
thDetection = 0.6

# データセットを学習用と評価用に分割する際の割合
splitRateTrain = 0.97

cate_name = ["background", "short sleeved shirt", "long sleeved shirt", "short sleeved outwear", "long sleeved outwear", "vest", "sling", "shorts", "trousers", "skirt", "short sleeved dress", "long sleeved dress", "vest dress", "sling dress"]
box_col = [(0, 0, 0), (192, 127, 0), (192, 0, 127), (0, 192, 127), (127, 192, 0), (127, 0, 192), (0, 127, 192), (255, 127, 0), (255, 0, 127), (0, 255, 127), (127, 255, 0), (127, 0, 255), (0, 127, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

def build_model(sw_train_eval):
    if sw_train_eval == "train":
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    else:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn()

    # 分類器にインプットする特徴量の数を取得
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # 事前訓練済みのヘッドを新しいものと置き換える
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, numClasses)

    # セグメンテーション・マスクの分類器に入力する特徴量の数を取得します
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    # セグメテーション・マスクの推論器を新しいものに置き換えます
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, numClasses)
    return model

if __name__ == "__main__":
    from torchsummary import summary
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = build_model("train")
    print(mdl)
    # summary(mdl, (3, imageSize, imageSize))
