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

img = cv.cvtColor(cv.imread(image_path), cv.COLOR_BGR2RGB) # 画像の読み込み・変換
data = cf.transforms_eval(img).unsqueeze(0).to(DEVICE) # テンソルに変換してから1次元追加

with torch.no_grad(): # 推定のために勾配計算の無効化モードで
    out0, out1 = model(data) # 推定処理

pred_0 = out0.item() * cf.val_rate_0 # 回帰の推定結果
pred_1_vals = out1[0].detach().cpu().numpy() # 識別の各クラスの推定値
pred_idx = torch.max(out1, 1)[1].item() # 識別の最大値のインデックス
print(pred_0, pred_1_vals, pred_idx)
