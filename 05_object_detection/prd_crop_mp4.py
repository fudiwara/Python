import sys
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

input_file_name = pathlib.Path(sys.argv[2]) # 入力のmp4ファイル
vc = cv2.VideoCapture(sys.argv[2])
crop_rect = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])

output_path = pathlib.Path(sys.argv[3]) # 出力先ディレクトリ
if(not output_path.exists()): output_path.mkdir()

img_output_dir = pathlib.Path("img_out")
if(not img_output_dir.exists()): img_output_dir.mkdir()

# モデルの定義と読み込みおよび評価用のモードにセットする
model = cf.build_model()
if DEVICE == "cuda":  model.load_state_dict(torch.load(model_path))
else: model.load_state_dict(torch.load(model_path, torch.device("cpu")))
model.to(DEVICE)
model.eval()

data_transforms = T.Compose([T.ToTensor()])

# フォントの設定
textsize = 16
linewidth = 3
font = ImageFont.truetype("_FiraMono-Medium.otf", size=textsize)

sw = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
sh = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
ssize = (sw, sh)
frame_count = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
frame_rate = int(vc.get(cv2.CAP_PROP_FPS))
print(ssize, frame_count, frame_rate, crop_rect)

dw, dh =  4 * (crop_rect[2] // 4), 4 * (crop_rect[3] // 4)
dsize = (dw, dh)
droi = (crop_rect[0], crop_rect[1], crop_rect[0] + dw, crop_rect[1] + dh)
fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # ファイル形式
output_file_name = str(output_path / input_file_name.stem) + "_dst.mp4"
vw = cv2.VideoWriter(output_file_name, fmt, frame_rate, dsize)
print(output_file_name, dsize)

for i in range(frame_count):
    ret, frame = vc.read()
    src_roi = frame[droi[1] : droi[3], droi[0] : droi[2]]
    src_img = cv2.cvtColor(src_roi, cv2.COLOR_BGR2RGB)
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
    flag_no_ext = True
    for i in range(len(scores)):
        b = bboxs[i]
        # print(b)
        prd_val = scores[i]
        if prd_val < cf.thDetection: continue # 閾値以下が出現した段階で終了
        else: flag_no_ext = False # オブジェクトが一つでも検出されたらフラグを除外する
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

    if flag_no_ext:
        img_output_name = str(img_output_dir / input_file_name.stem) + "_" + str(i).zfill(8) + ".png"
        cv2.imwrite(img_output_name, src_roi)

    dst_img = np.array(img, dtype=np.uint8)
    dst_img = cv2.cvtColor(dst_img, cv2.COLOR_RGB2BGR)

    vw.write(dst_img)

vc.release()
vw.release()
print("done", output_file_name)