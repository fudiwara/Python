# mediapipeの領域分割
# selfie_multiclass_256x256.tflite の処理対象：color_mapの内容
# https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter

import sys
sys.dont_write_bytecode = True
import cv2 as cv
import numpy as np
import mediapipe as mp

cap = cv.VideoCapture(int(sys.argv[1])) # キャプチャ開始(複数カメラはIDを追加)

# デフォルトのカメラ環境を用いる場合
# cw, ch = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
cw, ch = 640, 480
cap.set(cv.CAP_PROP_FRAME_WIDTH, cw)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, ch)
print(cw, ch)

options = mp.tasks.vision.ImageSegmenterOptions( # セグメンテーションのオプション
    base_options = mp.tasks.BaseOptions(model_asset_path = "models/selfie_multiclass_256x256.tflite"), # 人用
    # base_options = mp.tasks.BaseOptions(model_asset_path = "deeplab_v3.tflite"), # VOCカテゴリ用
    output_category_mask = True)
segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(options) # 推定器の初期化
num_categories = 6 # selfie_multiclass_256x256.tflite の場合は6
det_parts = [2, 3] # 顔と肌を検出する例

ret, frame = cap.read()

tick_meter = cv.TickMeter() # FPS表示用のタイマー
tick_meter.start() # 計測開始

while True:
    ret, frame = cap.read() # VideoCaptureから1フレーム読み込む

    if not ret: # フレームの読み込みに失敗したらループ終了
        break

    img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB) # MediaPipe用にRGBに
    mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = img_rgb) # MediaPipeのImageオブジェクトへ

    segmented_masks = segmenter.segment(mp_image) # セグメンテーション処理
    category_masks = segmented_masks.category_mask.numpy_view() # ndarrayに変換
    category_masks = np.squeeze(category_masks) # (H, W, 1) -> (H, W) に変換

    binary_mask = np.isin(category_masks, det_parts) # 特定のIDを持つマスクのみを作る

    img_mask = binary_mask.astype(np.uint8) * 255 # 表示用の2値画像

    tick_meter.stop() # 計測終了
    cv.putText(frame, f"{tick_meter.getFPS():.1f}", (5, 32), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
    tick_meter.reset() # 次フレーム用に時計をリセット
    tick_meter.start() # 計測開始

    cv.imshow("capture", frame) # 入力フレーム
    cv.imshow("segmented mask", img_mask) # マスクの表示
    key = cv.waitKey(1) & 0xFF # ループの更新とキー取得
    if key == 27: # Escキーだったら
        break # プログラムを終了