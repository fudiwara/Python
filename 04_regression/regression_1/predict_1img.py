import sys
sys.dont_write_bytecode = True
import torch
import cv2 as cv

import config as cf

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
model_path = sys.argv[1] # モデルのパス
image_path = sys.argv[2] # 推定する画像のパス

# モデルの定義と読み込みおよび評価用のモードにセットする
model = cf.build_model("eval").to(DEVICE)
model.load_state_dict(torch.load(model_path, map_location = DEVICE, weights_only = False))
model.eval()

# 画像の読み込み・変換
img = cv.cvtColor(cv.imread(image_path), cv.COLOR_BGR2RGB)
data = cf.transforms_eval(img).unsqueeze(0).to(DEVICE) # テンソルに変換してから1次元追加

with torch.no_grad(): # 推定のために勾配計算の無効化モードで
    outputs = model(data) # 推定処理

# 結果には正規化用の係数を乗算する
pred_val = outputs[0].item() * cf.val_rate
print(pred_val)