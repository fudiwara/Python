import sys, os
sys.dont_write_bytecode = True
import torch
import torchvision.transforms as T
import cv2 as cv
import numpy as np
from PIL import Image
import pathlib

import config as cf

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
model_path = sys.argv[1] # モデルのパス
image_path = sys.argv[2] # 入力画像のパス
file_name = pathlib.Path(sys.argv[2])
np.set_printoptions(precision = 3, suppress = True) # 指数表現をやめて小数点以下の桁数を指定する

# フォントと枠の設定
font_scale = cv.getFontScaleFromHeight(cv.FONT_HERSHEY_DUPLEX, 11, 1)
colors = [(255, 100, 0), (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
clr_num = len(colors)

# モデルの定義と読み込みおよび評価用のモードにセットする
model = cf.build_model("eval")
if DEVICE == "cuda": model.load_state_dict(torch.load(model_path, weights_only = False))
else: model.load_state_dict(torch.load(model_path, torch.device("cpu"), weights_only = False))
model.to(DEVICE)
model.eval()

data_transforms = T.Compose([T.ToTensor()])

# 画像の読み込み・変換
img = Image.open(image_path).convert('RGB') # カラー指定で開く
i_w, i_h = img.size
data = data_transforms(img)
data = data.unsqueeze(0) # テンソルに変換してから1次元追加

data = data.to(DEVICE)
outputs = model(data) # 推定処理
# print(outputs)
bboxs = outputs[0]["boxes"].detach().cpu().numpy()
scores = outputs[0]["scores"].detach().cpu().numpy()
labels = outputs[0]["labels"].detach().cpu().numpy()
# print(bboxs, scores, labels)

img = cv.cvtColor(np.array(img, dtype=np.uint8), cv.COLOR_RGB2BGR)
for i in range(len(scores)):
    b = bboxs[i] # 入力画像のスケールの座標
    # print(b)
    prd_val = scores[i]
    if prd_val < cf.thDetection: continue # 閾値以下は飛ばす
    prd_cls = labels[i]

    x0, y0 = int(b[0]), int(b[1])
    p0, p1 = (x0, y0), (int(b[2]), int(b[3]))
    print(prd_cls, prd_val, p0, p1)
    
    text = f" {prd_cls}  {prd_val:.3f} " # クラスと確率
    (t_w, t_h), baseline = cv.getTextSize(text, cv.FONT_HERSHEY_DUPLEX, font_scale, 1) # テキスト部の矩形サイズ取得
    cv.rectangle(img, p0, p1, colors[prd_cls % clr_num], thickness = 2) # 検出領域の矩形
    cv.rectangle(img, (x0, y0 - t_h), (x0 + t_w, y0), colors[prd_cls % clr_num], thickness = -1) # テキストの背景の矩形
    cv.putText(img, text, p0, cv.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), 1, cv.LINE_AA)

cv.imwrite(f"{file_name.stem}_det.png", img)
