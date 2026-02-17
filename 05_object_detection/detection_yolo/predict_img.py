import sys
sys.dont_write_bytecode = True
import pathlib
import cv2 as cv
from ultralytics import YOLO

model = YOLO(sys.argv[1]) # モデルの読み込み
image_path = pathlib.Path(sys.argv[2]) # 入力画像のパス
img = cv.imread(image_path)

res = model.predict(img, save = False, conf = 0.3)

bboxs = res[0].boxes.xyxy.detach().cpu().numpy() # xyxyの矩形情報
scores = res[0].boxes.conf.detach().cpu().numpy() # スコアを取得
classes = res[0].boxes.cls.detach().cpu().numpy() # クラスを取得

for i in range(len(bboxs)):
    x0, y0, x1, y1 = bboxs[i]
    print(classes[i], scores[i], x0, y0, x1, y1)
    p0 = (int(x0), int(y0))
    p1 = (int(x1), int(y1))
    cv.rectangle(img, p0, p1, (0, 255, 0), 2) # 検出結果の矩形描画

    info_text = f"{classes[i]} {scores[i]:.2f}"
    cv.putText(img, info_text, (int(x0), int(y0) - 6), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
cv.imwrite(f"{image_path.stem}_det.png", img)