import sys, os, time
sys.dont_write_bytecode = True
import torch
from torch import nn
import torchvision.transforms as T
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pathlib

import config as cf
import load_dataset_annot as ld

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
model_path = sys.argv[1] # モデルのパス
image_dir_path = sys.argv[2] # 入力画像が入っているディレクトリのパス
# image_path = sys.argv[2] # 入力画像のパス
# file_name = pathlib.Path(sys.argv[2])
output_dir = pathlib.Path(sys.argv[3]) # 画像を保存するフォルダ
if(not output_dir.exists()): output_dir.mkdir() # ディレクトリ生成
np.set_printoptions(precision=3, suppress=True) # 指数表現をやめて小数点以下の桁数を指定する

# フォントの設定
textsize = 16
linewidth = 3
font = ImageFont.truetype("_FiraMono-Medium.otf", size=textsize)

# モデルの定義と読み込みおよび評価用のモードにセットする
model = cf.build_model()
if DEVICE == "cuda":  model.load_state_dict(torch.load(model_path))
else: model.load_state_dict(torch.load(model_path, torch.device("cpu")))
model.to(DEVICE)
model.eval()

exts = ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG'] # 処理対象の拡張子
data_transforms = T.Compose([T.ToTensor()])

proc_time = []
fileList = list(pathlib.Path(image_dir_path).iterdir())
fileList.sort()
for i in range(len(fileList)):
    if fileList[i].is_file() and (fileList[i].suffix in exts): # ファイルのみ処理する
        s_tm = time.time()
        image_path = fileList[i]
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

        draw = ImageDraw.Draw(img)
        for i in range(len(scores)):
            b = bboxs[i]
            # print(b)
            prd_val = scores[i]
            if prd_val < cf.thDetection: break # 閾値以下が出現した段階で終了
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

        output_filename = f"{file_name.stem}_det.png"
        output_img_path = output_dir / output_filename
        img.save(output_img_path)
        proc_time.append((time.time() - s_tm))

proc_time = np.array(proc_time)
print(np.mean(proc_time))