import sys, time
sys.dont_write_bytecode = True
import torch
from torch import nn
import torchvision.transforms as T
import cv2
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import pathlib

import config as cf

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
model_path = sys.argv[1] # モデルのパス

cam_num = int(sys.argv[2]) # カメラのID
cap = cv2.VideoCapture(cam_num)

# モデルの定義と読み込みおよび評価用のモードにセットする
model = cf.build_model("eval")
if DEVICE == "cuda":  model.load_state_dict(torch.load(model_path))
else: model.load_state_dict(torch.load(model_path, torch.device("cpu")))
model.to(DEVICE)
model.eval()

data_transforms = T.Compose([T.ToTensor()])

# フォントの設定
textsize = 16
linewidth = 3
font = ImageFont.truetype("_FiraMono-Medium.otf", size=textsize)

sw, sh = 640, 480
ssize = (sw, sh)
dst_img = np.ones((sh, sw, 3), np.uint8) * 255 # 例外処理用に空の画像を作っておく

while True:
    s_tm = time.time()
    ret, frame = cap.read() # VideoCaptureから1フレーム読み込む

    if ret: # 読み込めた場合に処理をする
    # if False:
        src_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(src_img) # OpenCV形式からPIL形式へ変換
        data = data_transforms(img)
        data = data.unsqueeze(0) # テンソルに変換してから1次元追加

        data = data.to(DEVICE)
        outputs = model(data) # 推定処理
        # print(outputs)
        bboxs = outputs[0]["boxes"].detach().cpu().numpy()
        scores = outputs[0]["scores"].detach().cpu().numpy()
        labels = outputs[0]["labels"].detach().cpu().numpy()
        # print(bboxs, scores, labels)

        draw = ImageDraw.Draw(img)
        for i in range(len(scores)):
            b = bboxs[i]
            # print(b)
            prd_val = scores[i]
            if prd_val < cf.thDetection: continue # 閾値以下が出現した段階で終了
            prd_cls = labels[i]

            x0, y0 = b[0], b[1]
            p0 = (x0, y0)
            p1 = (b[2], b[3])
            print(prd_cls, prd_val, p0, p1)
            
            if prd_cls == 1: box_col = (255, 0, 0)
            else: box_col = (0, 255, 0)

            draw.rectangle((p0, p1), outline=box_col, width=linewidth) # 枠の矩形描画
            text = f" {prd_cls}  {prd_val:.3f} " # クラスと確率
            # txw, txh = draw.textsize(text, font=font) # 表示文字列のサイズ
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font) # 表示文字列のサイズ
            txw, txh = right - left, bottom - top
            txpos = (x0, y0 - textsize - linewidth // 2) # 表示位置
            draw.rectangle([txpos, (x0 + txw, y0)], outline=box_col, fill=box_col, width=linewidth)
            draw.text(txpos, text, font=font, fill=(255, 255, 255))

        dst_img = np.array(img, dtype=np.uint8)
        dst_img = cv2.cvtColor(dst_img, cv2.COLOR_RGB2BGR)
        
        fps_val = time.time() - s_tm
        cv2.putText(dst_img, f"{fps_val:.3f}", (5, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255,0), 1, cv2.LINE_AA)

    cv2.imshow("image", dst_img)

    # キー入力を1ms待って、kキーの場合(27 / ESC)だったらBreakする
    key = cv2.waitKey(1)
    if key == 27: break

