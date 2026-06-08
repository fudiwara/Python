import sys, pathlib
sys.dont_write_bytecode = True

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from ultralytics import YOLO

model_path = sys.argv[1] # モデルのパス
image_dir_path = pathlib.Path(sys.argv[2]) # 入力画像が入っているディレクトリのパス

model = YOLO(model_path) # 学習済みYOLO-CLSモデルを読み込み

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp"] # 処理対象の拡張子
img_paths = sorted([p for p in image_dir_path.glob("**/*") if p.suffix.lower() in IMG_EXTS])
label_list, pred_list = [], []
for i in range(len(img_paths)):
    results = model(img_paths[i], verbose=False) # 1枚の画像に対して推論
    parent_dir_name = img_paths[i].parent.name # 親ディレクトリ名を取得
    gt_cls = int(parent_dir_name.split("_")[0]) # 親ディレクトリ名から真値のクラスIDを取得

    pred_class_id = int(results[0].probs.top1) # Top-1のクラスIDを取得

    label_list.append(gt_cls)
    pred_list.append(pred_class_id)

    print(img_paths[i].name, gt_cls, pred_class_id) # ファイル名、真値、推定結果


print(accuracy_score(label_list, pred_list)) # 正解率
print(confusion_matrix(label_list, pred_list)) # 混同行列
print(classification_report(label_list, pred_list)) # 各種評価指標