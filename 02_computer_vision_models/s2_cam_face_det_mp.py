# MediaPipeの顔検出機能 (60ピクセルぐらいより大きいと快適に検出できる)
# blaze_face_short_range.tflite のダウンロードURL
# https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector

import sys
sys.dont_write_bytecode = True
import cv2 as cv
import mediapipe as mp

cap = cv.VideoCapture(int(sys.argv[1])) # VideoCaptureのインスタンス

# デフォルトのカメラ環境を用いる場合
# cw, ch = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
cw, ch = 640, 480
cap.set(cv.CAP_PROP_FRAME_WIDTH, cw)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, ch)
print(cw, ch)

options = mp.tasks.vision.FaceDetectorOptions(mp.tasks.BaseOptions("models/blaze_face_short_range.tflite")) # 検出器のオプション
detector = mp.tasks.vision.FaceDetector.create_from_options(options) # 検出器の初期化

tick_meter = cv.TickMeter() # FPS表示用のタイマー
tick_meter.start() # 計測開始

while True:
    ret, frame = cap.read() # VideoCaptureから1フレーム読み込む

    if not ret: # フレームの読み込みに失敗したらループ終了
        break

    img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB) # MediaPipe用にRGBに
    mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = img_rgb) # MediaPipeのImageオブジェクトへ

    detection_result = detector.detect(mp_image) # 顔検出

    if detection_result.detections:
        for i, detection in enumerate(detection_result.detections): # 各顔に対するループ
            b = detection.bounding_box
            # print(b)
            p0 = (b.origin_x, b.origin_y) # バウンディングボックスは入力画像の座標
            p1 = (b.origin_x + b.width, b.origin_y + b.height)
            cv.rectangle(frame, p0, p1, (0, 255, 0), 2) # 緑色の矩形

            score = int(detection.categories[0].score * 100) # 検出スコアを%で表示
            text = f"{i} {score:2d}%"
            cv.putText(frame, text, (b.origin_x, b.origin_y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            for keypoint in detection.keypoints: # キーポイントの描画
                keypoint_x = int(keypoint.x * cw) # キーポイントはUV座標系
                keypoint_y = int(keypoint.y * ch)
                cv.circle(frame, (keypoint_x, keypoint_y), 5, (0, 255, 0), 1) # 緑色の点

    tick_meter.stop() # 計測終了
    cv.putText(frame, f"{tick_meter.getFPS():.1f}", (5, 32), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
    tick_meter.reset() # 次フレーム用に時計をリセット
    tick_meter.start() # 計測開始

    cv.imshow("capture", frame) # 画像の描画
    key = cv.waitKey(1) & 0xFF # ループの更新とキー取得
    if key == 27: # Escキーだったら
        break # プログラムを終了