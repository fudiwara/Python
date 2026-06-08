import sys, pathlib
sys.dont_write_bytecode = True
import torch
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

fileList = sorted([p for p in image_dir_path.iterdir() if p.suffix.lower() in cf.ext])
for i in range(len(fileList)):
    image_path = fileList[i]

    # 画像の読み込み・変換
    img = cv.cvtColor(cv.imread(str(image_path)), cv.COLOR_BGR2RGB)
    data = cf.transforms_eval(img).unsqueeze(0).to(DEVICE) # テンソルに変換してから1次元追加

    with torch.no_grad(): # 推定のために勾配計算の無効化モードで
        outputs = model(data) # 推定処理

    # 結果には正規化用の係数を乗算する
    pred_val_0 = outputs[0][0].item() * cf.val_rate_0
    pred_val_1 = outputs[0][1].item() * cf.val_rate_1
    print(f"{pred_val_0:.3f} {pred_val_1:.3f} {image_path.name}")
