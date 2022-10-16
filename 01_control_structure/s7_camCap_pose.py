# mediapipeのposeお試しプログラム
# 鼻、左手、右手、左足、右足の場所に○を表示します
# 例外処理をしていないので体全体が全て写っているという前提で安定動作する感じです(入っていないパーツは荒ぶるカモ)

import sys
sys.dont_write_bytecode = True
import cv2
import mediapipe as mp

cap = cv2.VideoCapture(int(sys.argv[1])) # キャプチャ開始(複数カメラはIDを追加)

mp_pose = mp.solutions.pose # ポーズの初期化(変数はこのままでいいかな？)
pose = mp_pose.Pose(min_detection_confidence = 0.7, min_tracking_confidence = 0.5)

posName = [0, 19, 20, 31, 32] # 鼻、左手、右手、左足、右足
# https://google.github.io/mediapipe/solutions/pose.html

while True:
    ret, frame = cap.read() # キャプチャ
    if not ret: continue # キャプチャできていなければ再ループ
    cam_height, cam_width, _ = frame.shape # フレームサイズ取得(一回やればほんとうはいいのだけど)

    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) # mediapipeに処理を渡す
    if results.pose_landmarks is None: continue # 未検出なら再ループ

    pl = results.pose_landmarks
    for n in range(len(posName)): # 上記で設定したパーツ分ループする
        x = int(pl.landmark[posName[n]].x * cam_width)
        y = int(pl.landmark[posName[n]].y * cam_height)
        cv2.circle(frame, (x, y), 15, (0, 255, 0), thickness = 2)
        # print(x, y) # デバッグ用

    cv2.imshow("Frame pose", frame) # 円描画した結果の表示

    k = cv2.waitKey(1)
    if k == 27: break # escキーでプログラム終了

cap.release() # 後処理
cv2.destroyAllWindows()