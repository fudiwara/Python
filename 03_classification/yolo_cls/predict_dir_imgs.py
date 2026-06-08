import sys, pathlib
sys.dont_write_bytecode = True

from ultralytics import YOLO

model_path = sys.argv[1] # モデルのパス
image_dir_path = pathlib.Path(sys.argv[2]) # 入力画像が入っているディレクトリのパス

model = YOLO(model_path) # 学習済みYOLO-CLSモデルを読み込み

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp"] # 処理対象の拡張子
img_paths = sorted([p for p in image_dir_path.iterdir() if p.suffix.lower() in IMG_EXTS])
for i in range(len(img_paths)):
    results = model(img_paths[i], verbose=False) # 1枚の画像に対して推論

    pred_class_id = int(results[0].probs.top1) # Top-1のクラスIDを取得

    print(img_paths[i].name, pred_class_id) # ファイル名、推定結果