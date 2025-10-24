# mediapipeのface_detectionお試しプログラム
# 顔の矩形と特徴点の座標を表示します

import sys
sys.dont_write_bytecode = True
import cv2 as cv
import mediapipe as mp

cap = cv.VideoCapture(int(sys.argv[1])) # キャプチャ開始(複数カメラはIDを追加)

mp_face_detection = mp.solutions.face_detection # model_selectionは定義できないもよう
# face_detection = mp_face_detection.FaceDetection(model_selection = 1, min_detection_confidence = 0.5)
face_detection = mp_face_detection.FaceDetection(min_detection_confidence = 0.5)

while True:
    ret, frame = cap.read() # キャプチャ
    if not ret: continue # キャプチャできていなければ再ループ
    cam_height, cam_width, _ = frame.shape # フレームサイズ取得(一回やればほんとうはいいのだけど)

    results = face_detection.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB)) # mediapipeに処理を渡す
    if results.detections:
        for i in range(len(results.detections)):
            # 顔領域の検出結果描画
            b = results.detections[i].location_data.relative_bounding_box
            x0 = int(b.xmin * cam_width)
            y0 = int(b.ymin * cam_height)
            x1 = int((b.xmin + b.width) * cam_width)
            y1 = int((b.ymin + b.height) * cam_height)
            cv.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0))

            # 顔特徴点の検出結果描画
            k = results.detections[i].location_data.relative_keypoints
            for j in range(len(k)):
                kx = int(k[j].x * cam_width)
                ky = int(k[j].y * cam_height)
                cv.circle(frame, (kx, ky), 2, (0, 255, 0))
            # print(len(k))

    cv.imshow("image", frame) # 円描画した結果の表示

    k = cv.waitKey(1)
    if k == 27: break # escキーでプログラム終了
