import sys, os, time
sys.dont_write_bytecode = True
import torch
import torchvision.transforms as T
import cv2 as cv
import numpy as np
from PIL import Image
import pathlib

import config as cf
import load_dataset_annot as ld

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
model_path = sys.argv[1] # モデルのパス
image_dir_path = pathlib.Path(sys.argv[2]) # テスト画像が入っているディレクトリのパス
annotations_path = sys.argv[3] # アノテーションファイルへのフルパス(上のフォルダに依存しない)
output_dir = pathlib.Path(sys.argv[4]) # 画像を保存するフォルダ
if(not output_dir.exists()): output_dir.mkdir() # ディレクトリ生成
np.set_printoptions(precision=3, suppress=True) # 指数表現をやめて小数点以下の桁数を指定する

# フォントと枠の設定
font_scale = cv.getFontScaleFromHeight(cv.FONT_HERSHEY_DUPLEX, 11, 1)
colors = [(255, 100, 0), (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
clr_num = len(colors)

# アノテーションデータの読み込み
image_filenames = []
rect_info = []
with open(annotations_path, "r") as f:# self.filelines = f.read().split('\n')
    for line in f:
        l = line.split(" ")
        if 1 < len(l):
            rects = []
            image_filenames.append(l[0])
            for i in range(len(l) - 1):
                p = l[i + 1].split(",") # カンマ区切りのセットリスト
                x0 = float(p[0]) # 実数で読み込む
                y0 = float(p[1])
                x1 = float(p[2])
                y1 = float(p[3])
                c_num = int(p[4])
                rects.append([x0, y0, x1, y1, c_num])

            rect_info.append(rects)

def search_neighbourhood(x0, y0, x1, y1, ps): # x, yに一番近い点のIDを得る
    L = np.array([])
    x = (x0 + x1) / 2
    y = (y0 + y1) / 2
    for i in range(len(ps)):
        px = (ps[i][0] + ps[i][2]) / 2
        py = (ps[i][1] + ps[i][3]) / 2
        norm = np.sqrt( (px - x) ** 2 + (py - y) ** 2 )
        L = np.append(L, norm)
    return np.argmin(L)

def calc_iou(x0, y0, x1, y1, r):
    # print(x0, y0, x1, y1, r[0], r[1], r[2], r[3])
    # a, bは矩形を表すリストで、a=[xmin, ymin, xmax, ymax]
    ax_mn, ay_mn, ax_mx, ay_mx = x0, y0, x1, y1
    bx_mn, by_mn, bx_mx, by_mx = r[0], r[1], r[2], r[3]

    a_area = (ax_mx - ax_mn + 1) * (ay_mx - ay_mn + 1)
    b_area = (bx_mx - bx_mn + 1) * (by_mx - by_mn + 1)

    abx_mn = max(ax_mn, bx_mn)
    aby_mn = max(ay_mn, by_mn)
    abx_mx = min(ax_mx, bx_mx)
    aby_mx = min(ay_mx, by_mx)
    w = max(0, abx_mx - abx_mn + 1)
    h = max(0, aby_mx - aby_mn + 1)
    intersect = w * h

    iou = intersect / (a_area + b_area - intersect)
    return iou

def rect_comp_rectinfo(x0, y0, x1, y1, c_num, rect):
    # print(x0, y0, x1, y1)
    # print(rect)
    r_i = search_neighbourhood(x0, y0, x1, y1, rect)
    # print(r_i)
    val_iou = calc_iou(x0, y0, x1, y1, rect[r_i])
    return val_iou

# モデルの定義と読み込みおよび評価用のモードにセットする
model = cf.build_model("eval")
if DEVICE == "cuda": model.load_state_dict(torch.load(model_path))
else: model.load_state_dict(torch.load(model_path, torch.device("cpu")))
model.to(DEVICE)
model.eval()

data_transforms = T.Compose([T.ToTensor()])

proc_time = []
det_just_num, det_over_num, det_not_num, vals_iou = [], [], [], []
for idx in range(len(image_filenames)):
    file_name = image_dir_path / image_filenames[idx]
    print(file_name.name)

    # 画像の読み込み・変換
    img = Image.open(file_name).convert("RGB") # カラー指定で開く
    i_w, i_h = img.size
    data = data_transforms(img)
    data = data.unsqueeze(0) # テンソルに変換してから1次元追加
    s_tm = time.time()

    data = data.to(DEVICE)
    outputs = model(data) # 推定処理
    # print(outputs)
    bboxs = outputs[0]["boxes"].detach().cpu().numpy()
    scores = outputs[0]["scores"].detach().cpu().numpy()
    labels = outputs[0]["labels"].detach().cpu().numpy()
    # print(bboxs, scores, labels)

    img = cv.cvtColor(np.array(img, dtype=np.uint8), cv.COLOR_RGB2BGR)
    det_objs = 0
    for i in range(len(scores)):
        b = bboxs[i]
        # print(b)
        prd_val = scores[i]
        if prd_val < cf.thDetection: break # 閾値以下が出現した段階で終了
        prd_cls = labels[i]

        x0, y0 = int(b[0]), int(b[1])
        p0, p1 = (x0, y0), (int(b[2]), int(b[3]))
        print(prd_cls, prd_val, p0, p1)
        
        box_col = colors[prd_cls % clr_num]

        text = f" {prd_cls}  {prd_val:.3f} " # クラスと確率
        (t_w, t_h), baseline = cv.getTextSize(text, cv.FONT_HERSHEY_DUPLEX, font_scale, 1) # テキスト部の矩形サイズ取得
        cv.rectangle(img, p0, p1, box_col, thickness = 2) # 検出領域の矩形
        cv.rectangle(img, (x0, y0 - t_h), (x0 + t_w, y0), box_col, thickness = -1) # テキストの背景の矩形
        cv.putText(img, text, p0, cv.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), 1, cv.LINE_AA)

        val_iou = rect_comp_rectinfo(b[0], b[1], b[2], b[3], prd_cls, rect_info[idx])
        # print(val_iou)

        det_objs += 1
        vals_iou.append(val_iou)
    
    true_objs = len(rect_info[idx])
    
    if det_objs == true_objs:
        det_just_num.append(1)
        det_over_num.append(0)
        det_not_num.append(0)
    else:
        det_just_num.append(0)
        if det_objs < true_objs:
            det_not_num.append(true_objs - det_objs)
            det_over_num.append(0)
        else:
            det_not_num.append(0)
            det_over_num.append(det_objs - true_objs)
        

    output_filename = f"{file_name.stem}_det.png"
    output_img_path = output_dir / output_filename
    cv.imwrite(output_img_path, img)
    proc_time.append((time.time() - s_tm))

det_rate = np.sum(det_just_num) / len(image_filenames)
proc_time = np.array(proc_time)
print(np.mean(vals_iou), np.mean(proc_time))
print(det_rate, np.sum(det_over_num), np.sum(det_not_num))