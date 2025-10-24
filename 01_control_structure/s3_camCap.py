# カメラIDを指定する必要があるため
# python s3_camCap.py 0
# のように実行する

import sys
sys.dont_write_bytecode = True
import numpy as np
import cv2 as cv

cap = cv.VideoCapture(int(sys.argv[1])) # VideoCaptureのインスタンス
print(cap.isOpened())

# デフォルトのカメラ環境を用いる場合
# cw, ch = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
cw, ch = 360, 240

cap.set(cv.CAP_PROP_FRAME_WIDTH, cw)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, ch)
print(cw, ch)
img = np.ones((ch, cw, 3), np.uint8) * 255 # 例外処理用に空の画像を作っておく

while True:
    ret, frame = cap.read() # VideoCaptureから1フレーム読み込む
    # print("ret", ret)

    if ret: # 読み込めた場合に処理をする
        img = frame.copy()

        # 以下に取得した画像 img を用いた処理を記述する

        h, w, _ = frame.shape
        print(w, h)
    
    cv.imshow("image", img)

    # キー入力を1ms待って、kキーの場合(27 / ESC)だったらBreakする
    key = cv.waitKey(1)
    if key == 27: break
    elif key == ord("s"): cv.imwrite("save.jpg", img) # sキーの場合は画像保存する
