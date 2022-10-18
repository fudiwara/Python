import sys, os
sys.dont_write_bytecode = True

import torch
from torch import nn
import torchvision.transforms as T
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pathlib

import config as cf

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
# torch.backends.cudnn.benchmark = True
model_path = sys.argv[1] # モデルのパス
image_path = sys.argv[2] # 入力画像のパス
file_name = pathlib.Path(sys.argv[2])
np.set_printoptions(precision=3, suppress=True) # 指数表現をやめて小数点以下の桁数を指定する

model = cf.create_model(cf.numClasses, cf.imageSize, architecture=cf.modelArchitecture)
# model = nn.DataParallel(model) # 学習時とあわせてDataParallelを指定する

# モデルの定義と読み込みおよび評価用のモードにセットする
if DEVICE == "cuda":  model.load_state_dict(torch.load(model_path))
else: model.load_state_dict(torch.load(model_path, torch.device("cpu")))
model.to(DEVICE)
model.eval()

# フォントの設定
textsize = 16
linewidth = 3
font = ImageFont.truetype("font/FiraMono-Medium.otf", size=textsize)

# 画像の読み込み・変換
img = Image.open(image_path).convert('RGB') # カラー指定で開く
i_w, i_h = img.size
data_transforms = T.Compose([T.ToTensor()])
data = data_transforms(img.resize((cf.imageSize, cf.imageSize)))
data = data.unsqueeze(0) # テンソルに変換してから1次元追加
# print(data)
# print(data.shape)

num_images = data.shape[0]
dummy_targets = {
        "bbox": [
            torch.tensor([[0.0, 0.0, 0.0, 0.0]], device=DEVICE)
            for i in range(num_images)
        ],
        "cls": [torch.tensor([1.0], device=DEVICE) for i in range(num_images)],
        "img_size": torch.tensor(
            [(cf.imageSize, cf.imageSize)] * num_images, device=DEVICE
        ).float(),
        "img_scale": torch.ones(num_images, device=DEVICE).float(),
    }

data = data.to(DEVICE)
outputs = model(data, dummy_targets) # 推定処理
detections = outputs["detections"]
bboxs = detections.detach().cpu().numpy()

draw = ImageDraw.Draw(img)
for i in range(100):
    b = bboxs[0][i]
    print(b)
    prd_val = b[4]
    if prd_val < cf.thDetection: break # 閾値以下が出現した段階で終了
    prd_cls = int(b[5])
    x0, y0 = i_w * b[0] / cf.imageSize, i_h * b[1] / cf.imageSize
    p0 = (x0, y0)
    p1 = (i_w * b[2] / cf.imageSize, i_h * b[3] / cf.imageSize)
    print(prd_cls, prd_val, p0, p1)
    
    if prd_cls == 1: box_col = (255, 0, 0)
    else: box_col = (0, 255, 0)

    draw.rectangle((p0, p1), outline=box_col, width=linewidth) # 枠の矩形描画
    text = f" {prd_cls}  {prd_val:.3f} " # クラスと確率
    txw, txh = draw.textsize(text, font=font) # 表示文字列のサイズ
    txpos = (x0, y0 - textsize - linewidth // 2) # 表示位置
    draw.rectangle([txpos, (x0 + txw, y0)], outline=box_col, fill=box_col, width=linewidth)
    draw.text(txpos, text, font=font, fill=(255, 255, 255))

img.save(f"{file_name.stem}_det.png")
