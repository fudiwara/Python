# mediapipeのpose検出
# pose_landmarker_full.task のダウンロードURL
# https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker

import sys
sys.dont_write_bytecode = True
import cv2 as cv
import mediapipe as mp

cap = cv.VideoCapture(int(sys.argv[1])) # キャプチャ開始(複数カメラはIDを追加)

# デフォルトのカメラ環境を用いる場合
# cw, ch = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
cw, ch = 640, 480
cap.set(cv.CAP_PROP_FRAME_WIDTH, cw)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, ch)
print(cw, ch)

options = mp.tasks.vision.PoseLandmarkerOptions( # 検出器のオプション
    base_options = mp.tasks.BaseOptions(model_asset_path = "models/pose_landmarker_full.task"),
    # base_options = mp.tasks.BaseOptions(model_asset_path = "models/pose_landmarker_heavy.task"),
    num_poses = 2) # 検出できる人数
landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options) # 検出器の初期化

posName = [0, 19, 20, 31, 32] # 鼻、左手、右手、左足、右足
# https://google.github.io/mediapipe/solutions/pose.html

ret, frame = cap.read()
i_h, i_w, _ = frame.shape

tick_meter = cv.TickMeter() # FPS表示用のタイマー
tick_meter.start() # 計測開始

while True:
    ret, frame = cap.read() # VideoCaptureから1フレーム読み込む

    if not ret: # フレームの読み込みに失敗したらループ終了
        break

    img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB) # MediaPipe用にRGBに
    mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = img_rgb) # MediaPipeのImageオブジェクトへ

    pose_landmarker_result = landmarker.detect(mp_image) # poseランドマーク検出

    if pose_landmarker_result:
        for i, pose in enumerate(pose_landmarker_result.pose_landmarks): # 各人に対するループ

            for j, p in enumerate(pose): # 関節の各点に対するループ
                x = int(p.x * i_w)
                y = int(p.y * i_h)

                radius = int(11 - (p.z * 20)) # z座標に応じて半径を変える例
                radius = max(2, radius) # 半径が0以下にならないように

                cv.circle(frame, (x, y), radius, (0, 255, 0), 3) # 緑色の円で関節を表示


            # for n in range(len(posName)): # posNameで指定した関節の各点に対するループ
            #     x = int(pose[posName[n]].x * i_w)
            #     y = int(pose[posName[n]].y * i_h)
            #     cv.circle(frame, (x, y), 11, (0, 255, 0), 3) # 緑色の円で関節を表示

    tick_meter.stop() # 計測終了
    cv.putText(frame, f"{tick_meter.getFPS():.1f}", (5, 32), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
    tick_meter.reset() # 次フレーム用に時計をリセット
    tick_meter.start() # 計測開始

    cv.imshow("capture", frame) # 画像の描画
    key = cv.waitKey(1) & 0xFF # ループの更新とキー取得
    if key == 27: # Escキーだったら
        break # プログラムを終了