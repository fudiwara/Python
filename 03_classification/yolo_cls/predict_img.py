import sys
sys.dont_write_bytecode = True

from ultralytics import YOLO

model_path = sys.argv[1]
image_path = sys.argv[2]

model = YOLO(model_path) # 学習済みYOLO-CLSモデルを読み込み

results = model(image_path, verbose=False) # 1枚の画像に対して推論

pred_class_id = int(results[0].probs.top1) # Top-1のクラスIDを取得
print(pred_class_id)
