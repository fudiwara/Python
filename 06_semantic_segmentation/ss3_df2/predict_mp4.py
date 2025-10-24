import sys, time, pathlib
sys.dont_write_bytecode = True

import cv2 as cv
import numpy as np
from PIL import Image

import torch
import torchvision

import config as cf

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
model_path = sys.argv[1] # モデルのパス
input_file_name = pathlib.Path(sys.argv[2]) # 入力のmp4ファイル
vc = cv.VideoCapture(sys.argv[2])

# フォントの設定
font_scale = cv.getFontScaleFromHeight(cv.FONT_HERSHEY_DUPLEX, 11, 1)

# モデルの定義と読み込みおよび評価用のモードにセットする
model = cf.build_model("eval")
if DEVICE == "cuda": model.load_state_dict(torch.load(model_path))
else: model.load_state_dict(torch.load(model_path, torch.device("cpu")))
model.to(DEVICE)
model.eval()

data_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

sw = int(vc.get(cv.CAP_PROP_FRAME_WIDTH))
sh = int(vc.get(cv.CAP_PROP_FRAME_HEIGHT))
ssize = (sw, sh)
frame_count = int(vc.get(cv.CAP_PROP_FRAME_COUNT))
frame_rate = int(vc.get(cv.CAP_PROP_FPS))
print(ssize, frame_count, frame_rate)

fmt = cv.VideoWriter_fourcc(*"mp4v") # ファイル形式
output_file_name = input_file_name.stem + "_dst.mp4"
vw = cv.VideoWriter(output_file_name, fmt, frame_rate, ssize)
print(output_file_name)

proc_time = []
for f in range(frame_count):
    s_tm = time.time()
    ret, frame = vc.read()
    if not ret: continue

    src_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
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

    for i in range(len(scores)):
        b = bboxs[i]
        # print(b)
        prd_val = scores[i]
        if prd_val < cf.thDetection: continue # 閾値以下が出現した段階で終了
        prd_cls = labels[i]

        x0, y0 = int(b[0]), int(b[1])
        p0, p1 = (x0, y0), (int(b[2]), int(b[3]))
        print(prd_cls, prd_val, p0, p1)

        text = f" {prd_cls}  {prd_val:.3f} " # クラスと確率
        (t_w, t_h), baseline = cv.getTextSize(text, cv.FONT_HERSHEY_DUPLEX, font_scale, 1) # テキスト部の矩形サイズ取得
        cv.rectangle(frame, p0, p1, cf.box_col[prd_cls], thickness = 2) # 検出領域の矩形
        cv.rectangle(frame, (x0, y0 - t_h), (x0 + t_w, y0), cf.box_col[prd_cls], thickness = -1) # テキストの背景の矩形
        cv.putText(frame, text, p0, cv.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), 1, cv.LINE_AA)
    
    vw.write(frame)
    proc_time.append((time.time() - s_tm))

vc.release()
vw.release()
proc_time = np.array(proc_time)
s_t = np.sum(proc_time)
m_t = np.mean(proc_time)
print(f"{output_file_name} total: {s_t}s,  average: {m_t:.3f}")

