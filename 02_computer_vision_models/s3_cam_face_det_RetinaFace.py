# RetinaFace (ResNet50) の顔検出
# pip install insightface onnxruntime
# 初回実行時にモデルが自動ダウンロードされる

import sys
sys.dont_write_bytecode = True
import cv2 as cv
from insightface.app import FaceAnalysis

cap = cv.VideoCapture(int(sys.argv[1])) # VideoCaptureのインスタンス

# デフォルトのカメラ環境を用いる場合
# cw, ch = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
cw, ch = 640, 480
cap.set(cv.CAP_PROP_FRAME_WIDTH, cw)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, ch)
print(cw, ch)

app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.6)

tick_meter = cv.TickMeter() # FPS表示用のタイマー
tick_meter.start() # 計測開始

while True:
    ret, frame = cap.read() # VideoCaptureから1フレーム読み込む

    if not ret: # フレームの読み込みに失敗したらループ終了
        break

    faces = app.get(frame) # 顔検出

    for i, face in enumerate(faces):
        x1, y1, x2, y2 = face.bbox.astype(int) # bbox: [x1, y1, x2, y2]
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        score = int(face.det_score * 100)
        text = f"{i} {score:2d}%"
        cv.putText(frame, text, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if face.kps is not None: # 目・鼻・口角の5点ランドマークを描画
            for (kx, ky) in face.kps.astype(int):
                cv.circle(frame, (kx, ky), 2, (0, 255, 0), 2)

    tick_meter.stop() # 計測終了
    cv.putText(frame, f"{tick_meter.getFPS():.1f}", (5, 32), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
    tick_meter.reset() # 次フレーム用に時計をリセット
    tick_meter.start() # 計測開始

    cv.imshow("capture", frame) # 画像の描画
    key = cv.waitKey(1) & 0xFF # ループの更新とキー取得
    if key == 27: # Escキーだったら
        break # プログラムを終了