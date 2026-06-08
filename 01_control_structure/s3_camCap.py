# カメラIDを指定する必要があるため
# python s3_camCap.py 0
# のように実行する

import sys
sys.dont_write_bytecode = True
import cv2 as cv

cap = cv.VideoCapture(int(sys.argv[1])) # VideoCaptureのインスタンス
print(cap.isOpened())

# デフォルトのカメラ環境を用いる場合
# cw, ch = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
cw, ch = 640, 480

cap.set(cv.CAP_PROP_FRAME_WIDTH, cw)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, ch)
print(cw, ch)

while True:
    ret, frame = cap.read() # VideoCaptureから1フレーム読み込む
    # print("ret", ret)

    if not ret: # フレームの読み込みに失敗したらループ終了
        break

    # 以下に取得した画像 img を用いた処理を記述する

    h, w, _ = frame.shape
    print(w, h)
    
    cv.imshow("image", frame)

    key = cv.waitKey(1) & 0xFF # ループの更新とキー取得
    if key == 27: # Escキーだったら
        break # プログラムを終了
    elif key == ord("s"): # sキーだったら
        cv.imwrite("save.jpg", frame) # 画像を保存する
