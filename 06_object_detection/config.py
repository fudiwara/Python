import sys
sys.dont_write_bytecode = True

import torch
from effdet.config.model_config import efficientdet_model_param_dict
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet

# 画像の一辺のサイズ (この大きさにリサイズされるので要確認)
imageSize = 512

# 繰り返す回数
epochSize = 100

# 学習時のバッチのサイズ
batchSize = 8

# カテゴリの総数 (背景の0を含めた合計)
numClasses = 2

# ネットワークの構成モデル名
modelArchitecture = "tf_efficientnetv2_m"

# 検出の閾値
thDetection = 0.6

# 学習の収束率
learningRate = 0.0002

# データセットを学習用と評価用に分割する際の割合
splitRateTrain = 0.8

def create_model(num_classes=1, image_size=512, architecture="tf_efficientnetv2_l"):
    efficientdet_model_param_dict[architecture] = dict(
        name=architecture,
        backbone_name=architecture,
        backbone_args=dict(drop_path_rate=0.2),
        num_classes=num_classes,
        url='', )
    
    config = get_efficientdet_config(architecture)
    config.update({'num_classes': num_classes})
    config.update({'image_size': (image_size, image_size)})
    # print(config)

    net = EfficientDet(config, pretrained_backbone=True)
    net.class_net = HeadNet(config, num_outputs=config.num_classes,)
    return DetBenchTrain(net, config)

if __name__ == "__main__":
    from torchsummary import summary
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = create_model(architecture=modelArchitecture)
    print(mdl)
    # summary(mdl, (3, imageSize, imageSize))
