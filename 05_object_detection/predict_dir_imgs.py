import sys, os, time
sys.dont_write_bytecode = True
import torch
from torch import nn
import torchvision.transforms as T
import cv2
import numpy as np
from PIL import Image
import pathlib

import config as cf
import load_dataset_annot as ld

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
model_path = sys.argv[1] # モデルのパス
image_dir_path = pathlib.Path(sys.argv[2]) # 入力画像が入っているディレクトリのパス
output_dir = pathlib.Path(sys.argv[3]) # 画像を保存するフォルダ
if(not output_dir.exists()): output_dir.mkdir() # ディレクトリ生成
np.set_printoptions(precision=3, suppress=True) # 指数表現をやめて小数点以下の桁数を指定する

# フォントの設定
font_scale = cv2.getFontScaleFromHeight(cv2.FONT_HERSHEY_DUPLEX, 11, 1)

# モデルの定義と読み込みおよび評価用のモードにセットする
model = cf.build_model("eval")
if DEVICE == "cuda":  model.load_state_dict(torch.load(model_path))
else: model.load_state_dict(torch.load(model_path, torch.device("cpu")))
model.to(DEVICE)
model.eval()

exts = [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"] # 処理対象の拡張子
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
clr_num = len(colors)
data_transforms = T.Compose([T.ToTensor()])

proc_time = []
fileList = sorted([p for p in image_dir_path.glob("**/*") if p.suffix in exts])
for f in range(len(fileList)):
    s_tm = time.time()
    image_path = fileList[f]
    file_name = pathlib.Path(image_path)

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

    img = cv2.cvtColor(np.array(img, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    for i in range(len(scores)):
        b = bboxs[i]
        # print(b)
        prd_val = scores[i]
        if prd_val < cf.thDetection: break # 閾値以下が出現した段階で終了
        prd_cls = labels[i]

        x0, y0 = int(b[0]), int(b[1])
        p0, p1 = (x0, y0), (int(b[2]), int(b[3]))
        print(image_path.stem, prd_cls, prd_val, p0, p1)
        
        text = f" {prd_cls}  {prd_val:.3f} " # クラスと確率
        (t_w, t_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, 1) # テキスト部の矩形サイズ取得
        cv2.rectangle(img, p0, p1, colors[prd_cls % clr_num], thickness = 2) # 検出領域の矩形
        cv2.rectangle(img, (x0, y0 - t_h), (x0 + t_w, y0), colors[prd_cls % clr_num], thickness = -1) # テキストの背景の矩形
        cv2.putText(img, text, p0, cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    output_filename = f"{file_name.stem}_det.png"
    output_img_path = output_dir / output_filename
    cv2.imwrite(str(output_img_path), img)
    proc_time.append((time.time() - s_tm))

proc_time = np.array(proc_time)
print(np.mean(proc_time))