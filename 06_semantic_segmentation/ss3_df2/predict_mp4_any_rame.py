import sys, pathlib
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
interval_min = int(sys.argv[3]) # フレームを確認する間隔(分)

# output_path = pathlib.Path(sys.argv[4]) # 出力先ディレクトリ
# if(not output_path.exists()): output_path.mkdir()

# フォントの設定
font_scale = cv.getFontScaleFromHeight(cv.FONT_HERSHEY_DUPLEX, 11, 1)

# モデルの定義と読み込みおよび評価用のモードにセットする
model = cf.build_model("eval")
if DEVICE == "cuda": model.load_state_dict(torch.load(model_path))
else: model.load_state_dict(torch.load(model_path, torch.device("cpu")))
model.to(DEVICE)
model.eval()

data_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

frame_count = int(vc.get(cv.CAP_PROP_FRAME_COUNT))
interval_frame = interval_min * 60 * 30 # 分 秒 フレーム

cls_log = [0] * cf.numClasses

for f in range(0, frame_count, interval_frame):
    vc.set(cv.CAP_PROP_POS_FRAMES, f) 
    ret, frame = vc.read()
    if not ret: continue

    src_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    img = Image.fromarray(src_img) # OpenCV形式からPIL形式へ変換
    data = data_transforms(img).unsqueeze(0).to(DEVICE) # テンソルに変換してから1次元追加
    outputs = model(data) # 推定処理
    # print(outputs)
    bboxs = outputs[0]["boxes"].detach().cpu().numpy()
    scores = outputs[0]["scores"].detach().cpu().numpy()
    labels = outputs[0]["labels"].detach().cpu().numpy()

    for i in range(len(scores)):
        b = bboxs[i]
        prd_val = scores[i]
        if prd_val < cf.thDetection: continue # 閾値以下が出現した段階で終了
        prd_cls = labels[i]
        cls_log[prd_cls] += 1

        x0, y0 = int(b[0]), int(b[1])
        p0, p1 = (x0, y0), (int(b[2]), int(b[3]))
        print(f, prd_cls, prd_val, p0, p1)

        text = f" {prd_cls}  {prd_val:.3f} " # クラスと確率
        (t_w, t_h), baseline = cv.getTextSize(text, cv.FONT_HERSHEY_DUPLEX, font_scale, 1) # テキスト部の矩形サイズ取得
        cv.rectangle(frame, p0, p1, cf.box_col[prd_cls], thickness = 2) # 検出領域の矩形
        cv.rectangle(frame, (x0, y0 - t_h), (x0 + t_w, y0), cf.box_col[prd_cls], thickness = -1) # テキストの背景の矩形
        cv.putText(frame, text, p0, cv.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), 1, cv.LINE_AA)
    
    # cv.imwrite(str(output_path / f"c{f:06}.jpg"), frame)

vc.release()

with open(f"{input_file_name.name}_log.csv", mode = "w") as f: # 各クラスの検出数を出力
    for i in range(cf.numClasses):
        print(f"{cf.cate_name[i]},{cls_log[i]}", file = f)