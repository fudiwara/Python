import sys, time
sys.dont_write_bytecode = True
import cv2
from PIL import Image
import numpy as np
import pathlib

import torch
import torchvision

import config as cf

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
model_path = sys.argv[1] # モデルのパス

input_file_name = pathlib.Path(sys.argv[2]) # 入力のmp4ファイル
vc = cv2.VideoCapture(sys.argv[2])

output_path = pathlib.Path(sys.argv[3]) # 出力先ディレクトリ
if(not output_path.exists()): output_path.mkdir()

# モデルの定義と読み込みおよび評価用のモードにセットする
model = cf.build_model("eval")
if DEVICE == "cuda": model.load_state_dict(torch.load(model_path))
else: model.load_state_dict(torch.load(model_path, torch.device("cpu")))
model.to(DEVICE)
model.eval()

data_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

# フォント、枠の設定
font_scale = cv2.getFontScaleFromHeight(cv2.FONT_HERSHEY_DUPLEX, 11, 1)
colors = [(255, 100, 0), (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
clr_num = len(colors)

sw = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
sh = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
ssize = (sw, sh)
frame_count = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
frame_rate = int(vc.get(cv2.CAP_PROP_FPS))
print(ssize, frame_count, frame_rate)

fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # ファイル形式
output_file_name = str(output_path / input_file_name.stem) + "_dst.mp4"
vw = cv2.VideoWriter(output_file_name, fmt, frame_rate, ssize)
print(output_file_name)

proc_time = []
for f in range(frame_count):
    s_tm = time.time()
    ret, frame = vc.read()
    if not ret: continue
    src_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(src_img) # OpenCV形式からPIL形式へ変換
    data = data_transforms(img).unsqueeze(0) # テンソルに変換してから1次元追加
    data = data.to(DEVICE)
    outputs = model(data) # 推定処理
    # print(outputs)
    bboxs = outputs[0]["boxes"].detach().cpu().numpy()
    scores = outputs[0]["scores"].detach().cpu().numpy()
    labels = outputs[0]["labels"].detach().cpu().numpy()
    # print(bboxs, scores, labels)

    flag_no_ext = True
    for i in range(len(scores)):
        b = bboxs[i]
        # print(b)
        prd_val = scores[i]
        if prd_val < cf.thDetection: continue # 閾値以下が出現した段階で終了
        else: flag_no_ext = False # オブジェクトが一つでも検出されたらフラグを除外する
        prd_cls = labels[i]

        x0, y0 = int(b[0]), int(b[1])
        p0, p1 = (x0, y0), (int(b[2]), int(b[3]))
        print(prd_cls, prd_val, p0, p1)
        
        box_col = colors[prd_cls % clr_num]

        text = f" {prd_cls}  {prd_val:.3f} " # クラスと確率
        (t_w, t_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, 1) # テキスト部の矩形サイズ取得
        cv2.rectangle(frame, p0, p1, box_col, thickness = 2) # テキストの背景の矩形
        cv2.rectangle(frame, (x0, y0 - t_h), (x0 + t_w, y0), box_col, thickness = -1) # 検出領域の矩形
        cv2.putText(frame, text, p0, cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    
    if flag_no_ext:
        cv2.imwrite(f"f{f:06}.png", frame)
    
    vw.write(frame)
    proc_time.append((time.time() - s_tm))

vc.release()
vw.release()
print("done", output_file_name)
proc_time = np.array(proc_time)
s_t = np.sum(proc_time)
m_t = np.mean(proc_time)
with open("_movie_log.txt", mode = "a") as f: f.write(f"{s_t}s {m_t:.3f} {output_file_name}\n")
