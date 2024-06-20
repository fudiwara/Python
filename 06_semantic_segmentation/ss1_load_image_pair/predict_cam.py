import sys, time
sys.dont_write_bytecode = True

import cv2
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T

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
font_scale = cv2.getFontScaleFromHeight(cv2.FONT_HERSHEY_DUPLEX, 11, 1)

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

            x0, y0 = int(b[0]), int(b[1])
            p0, p1 = (x0, y0), (int(b[2]), int(b[3]))
            print(prd_cls, prd_val, p0, p1)

            text = f" {prd_cls}  {prd_val:.3f} " # クラスと確率
            (t_w, t_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, 1) # テキスト部の矩形サイズ取得
            cv2.rectangle(frame, p0, p1, cf.box_col[prd_cls], thickness = 2) # 検出領域の矩形
            cv2.rectangle(frame, (x0, y0 - t_h), (x0 + t_w, y0), cf.box_col[prd_cls], thickness = -1) # テキストの背景の矩形
            cv2.putText(frame, text, p0, cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        
        fps_val = time.time() - s_tm
        cv2.putText(frame, f"{fps_val:.3f}", (5, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255,0), 1, cv2.LINE_AA)

    cv2.imshow("image", frame)

    # キー入力を1ms待って、kキーの場合(27 / ESC)だったらBreakする
    key = cv2.waitKey(1)
    if key == 27: break

