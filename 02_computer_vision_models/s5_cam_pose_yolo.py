# YOLOのpose検出
# pip install ultralytics
# 初回実行時にモデルが自動ダウンロードされる

import sys
sys.dont_write_bytecode = True
import cv2 as cv
from ultralytics import YOLO

cap = cv.VideoCapture(int(sys.argv[1])) # キャプチャ開始(複数カメラはIDを追加)

# デフォルトのカメラ環境を用いる場合
# cw, ch = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
cw, ch = 640, 480
cap.set(cv.CAP_PROP_FRAME_WIDTH, cw)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, ch)
print(cw, ch)

model = YOLO("yolo26s-pose.pt") # モデルのロード

posName = [0, 9, 10, 15, 16] # 鼻、左手、右手、左足、右足
# https://docs.ultralytics.com/tasks/pose#models

ret, frame = cap.read()
i_h, i_w, _ = frame.shape

tick_meter = cv.TickMeter() # FPS表示用のタイマー
tick_meter.start() # 計測開始

while True:
    ret, frame = cap.read() # VideoCaptureから1フレーム読み込む

    if not ret: # フレームの読み込みに失敗したらループ終了
        break

    results = model(frame) # モデルによる推論

    for res in results:
        if res.keypoints is None or len(res.keypoints.xy) == 0:
            continue

        xy = res.keypoints.xy[0] # 関節の座標
        print(xy)

        # for i, (x, y) in enumerate(xy): # 関節の各点に対するループ
        #     x, y = int(x), int(y)
        #     cv.circle(frame, (x, y), 11, (0, 255, 0), 3) # 緑色の円で関節を表示

        for n in range(len(posName)): # posNameで指定した関節の各点に対するループ
            x = int(xy[posName[n]][0])
            y = int(xy[posName[n]][1])
            cv.circle(frame, (x, y), 11, (0, 255, 0), 3) # 緑色の円で関節を表示

    tick_meter.stop() # 計測終了
    cv.putText(frame, f"{tick_meter.getFPS():.1f}", (5, 32), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
    tick_meter.reset() # 次フレーム用に時計をリセット
    tick_meter.start() # 計測開始

    cv.imshow("capture", frame) # 画像の描画
    key = cv.waitKey(1) & 0xFF # ループの更新とキー取得
    if key == 27: # Escキーだったら
        break # プログラムを終了