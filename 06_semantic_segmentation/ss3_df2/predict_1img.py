import sys, os
sys.dont_write_bytecode = True
import pathlib

import cv2 as cv
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T

import config as cf

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
model_path = sys.argv[1] # モデルのパス
image_path = sys.argv[2] # 入力画像のパス
file_name = pathlib.Path(image_path)
np.set_printoptions(precision = 3, suppress = True) # 指数表現をやめて小数点以下の桁数を指定する

# フォントの設定
font_scale = cv.getFontScaleFromHeight(cv.FONT_HERSHEY_DUPLEX, 11, 1)

# モデルの定義と読み込みおよび評価用のモードにセットする
model = cf.build_model("eval")
if DEVICE == "cuda": model.load_state_dict(torch.load(model_path))
else: model.load_state_dict(torch.load(model_path, torch.device("cpu")))
model.to(DEVICE)
model.eval()

data_transforms = T.Compose([T.ToTensor()])

# 画像の読み込み・変換
img = Image.open(image_path).convert("RGB") # カラー指定で開く
data = data_transforms(img)
data = data.unsqueeze(0) # テンソルに変換してから1次元追加

data = data.to(DEVICE)
outputs = model(data) # 推定処理
# print(outputs)
bboxs = outputs[0]["boxes"].detach().cpu().numpy()
scores = outputs[0]["scores"].detach().cpu().numpy()
labels = outputs[0]["labels"].detach().cpu().numpy()
masks = outputs[0]["masks"].detach().cpu().numpy()
print(masks.shape)
# print(bboxs, scores, labels)
# print(len(scores))
img = cv.cvtColor(np.array(img, dtype=np.uint8), cv.COLOR_RGB2BGR)
for i in range(len(scores)):
    b = bboxs[i]
    # print(b)
    prd_val = scores[i]
    if prd_val < cf.thDetection: continue # 閾値以下は飛ばす
    prd_cls = labels[i]

    x0, y0 = int(b[0]), int(b[1])
    p0, p1 = (x0, y0), (int(b[2]), int(b[3]))
    print(prd_cls, prd_val, p0, p1)

    cv.rectangle(img, p0, p1, cf.box_col[prd_cls], thickness = 2) # 検出領域の矩形
    text_parts = cf.cate_name[prd_cls]
    (t_w, t_h), baseline = cv.getTextSize(text_parts, cv.FONT_HERSHEY_DUPLEX, font_scale, 1) # テキスト部の矩形サイズ取得
    cv.rectangle(img, (x0, y0 - t_h), (x0 + t_w, y0), cf.box_col[prd_cls], thickness = -1) # テキストの背景の矩形
    cv.putText(img, text_parts, p0, cv.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), 1, cv.LINE_AA)
    text_val = f"{prd_val:.2f} " # クラスと確率
    (t_w, t_h), baseline = cv.getTextSize(text_val, cv.FONT_HERSHEY_DUPLEX, font_scale, 1) # テキスト部の矩形サイズ取得
    cv.rectangle(img, (x0, y0 + 1), (x0 + t_w, y0 + t_h + 1), cf.box_col[prd_cls], thickness = -1) # テキストの背景の矩形
    cv.putText(img, text_val, (x0, y0 + 11), cv.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), 1, cv.LINE_AA)

    # print(masks[i])
    # msk_img = masks[i][0]
    # msk_img = (msk_img * 255).astype(np.uint8)
    # outputFIlename = f"{file_name.stem}_{i}.png"
    # cv.imwrite(outputFIlename, msk_img) # マスク画像の保存

cv.imwrite(f"{file_name.stem}_det.png", img)
