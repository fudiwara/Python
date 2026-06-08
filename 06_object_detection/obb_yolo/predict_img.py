import sys
sys.dont_write_bytecode = True
import pathlib
import cv2 as cv
from ultralytics import YOLO

model = YOLO(sys.argv[1]) # モデルの読み込み
image_path = pathlib.Path(sys.argv[2]) # 入力画像のパス
img = cv.imread(image_path)

res = model.predict(img, save = False, conf = 0.3)

polys = res[0].obb.xyxyxyxy.cpu().numpy() # ポリゴンの座標
scores = res[0].obb.conf.detach().cpu().numpy() # スコアを取得

for i in range(len(polys)):
    pts = polys[i].reshape(-1, 2).astype(int)
    cv.polylines(img, [pts], isClosed = True, color = (0, 255, 0), thickness = 2) # 検出結果のポリゴン描画
    print(scores[i], pts)

cv.imwrite(f"{image_path.stem}_obb.png", img)