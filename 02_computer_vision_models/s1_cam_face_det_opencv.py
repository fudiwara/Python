# OpenCVの顔検出機能
# face_detection_yunet_2023mar.onnx のダウンロードURL
# https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

import sys
sys.dont_write_bytecode = True
import cv2 as cv

cap = cv.VideoCapture(int(sys.argv[1])) # VideoCaptureのインスタンス

# デフォルトのカメラ環境を用いる場合
# cw, ch = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
cw, ch = 640, 480
cap.set(cv.CAP_PROP_FRAME_WIDTH, cw)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, ch)
print(cw, ch)

model_path = "models/face_detection_yunet_2023mar.onnx" # 顔検出用モデル
detector = cv.FaceDetectorYN.create(model_path, "", (cw, ch), 0.6) # 顔検出機能の初期化

tick_meter = cv.TickMeter() # FPS表示用のタイマー
tick_meter.start() # 計測開始

while True:
    ret, frame = cap.read() # VideoCaptureから1フレーム読み込む

    if not ret: # フレームの読み込みに失敗したらループ終了
        break

    _, faces = detector.detect(frame) # 顔検出

    if faces is not None: # 顔検出がされたら
        for face in faces: # 検出された数の顔でループ
            x, y, w, h  = face[ : 4].astype(int) # 顔の左上XY座標と横幅、縦幅
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    tick_meter.stop() # 計測終了
    cv.putText(frame, f"{tick_meter.getFPS():.1f}", (5, 32), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
    tick_meter.reset() # 次フレーム用に時計をリセット
    tick_meter.start() # 計測開始

    cv.imshow("capture", frame) # 画像の描画
    key = cv.waitKey(1) & 0xFF # ループの更新とキー取得
    if key == 27: # Escキーだったら
        break # プログラムを終了