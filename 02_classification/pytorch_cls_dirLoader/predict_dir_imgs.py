import sys, pathlib
sys.dont_write_bytecode = True
import torch
import numpy as np
import cv2 as cv

import config as cf

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
model_path = sys.argv[1] # モデルのパス
image_dir_path = pathlib.Path(sys.argv[2]) # 入力画像が入っているディレクトリのパス

# モデルの定義と読み込みおよび評価用のモードにセットする
model = cf.build_model("eval").to(DEVICE)
model.load_state_dict(torch.load(model_path, map_location = DEVICE, weights_only = False))
model.eval()

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp"] # 処理対象の拡張子
img_paths = sorted([p for p in image_dir_path.iterdir() if p.suffix.lower() in IMG_EXTS])
for i in range(len(img_paths)):
    image_path = img_paths[i]
    img = cv.imread(str(image_path))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    data = cf.transforms_eval(img).unsqueeze(0).to(DEVICE) # テンソルに変換してから1次元追加
    # print(data)

    with torch.no_grad(): # 推定のために勾配計算の無効化モードで
        outputs = model(data) # 推定処理
    _, preds = torch.max(outputs, 1) # 1次元目の中の最大値を得る(最大値と最大値のインデックス)
    pred_idx = preds[0].cpu().numpy().tolist() # tensorから数値へ
    
    np.set_printoptions(precision = 3)
    pred_val = outputs[0].to("cpu").detach().numpy().copy() # 各クラスの推定値

    print(image_path.name, pred_idx) # ファイル名 と 結果のラベル
    print(pred_val) # 各カテゴリに対する推定結果の生データ
