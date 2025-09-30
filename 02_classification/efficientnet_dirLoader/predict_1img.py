import sys
sys.dont_write_bytecode = True
import torch
import torchvision.transforms as T
import numpy as np
import config as cf
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
model_path = sys.argv[1] # モデルのパス
image_path = sys.argv[2] # 推定する画像のパス

# モデルの定義と読み込みおよび評価用のモードにセットする
model = cf.build_model("eval").to(DEVICE)
if DEVICE == "cuda": model.load_state_dict(torch.load(model_path, weights_only = False))
else: model.load_state_dict(torch.load(model_path, torch.device("cpu")))
model.eval()

# 画像の読み込み・変換
img = Image.open(image_path).convert("RGB") # カラー指定で開く
data_transforms = T.Compose([T.Resize(cf.cellSize), T.CenterCrop(cf.cellSize), T.ToTensor()])
data = data_transforms(img).unsqueeze(0) # テンソルに変換してから1次元追加
# print(data.shape)

# 推定処理
data = data.to(DEVICE)
outputs = model(data)
_, preds = torch.max(outputs, 1) # 1次元目の中の最大値を得る(最大値と最大値のインデックス)
pred_idx = preds[0].cpu().numpy().tolist() # tensorから数値へ

# 結果の表示
np.set_printoptions(precision = 3)
pred_val = outputs[0].to("cpu").detach().numpy().copy() # 各クラスの推定値
print(pred_val)
print(pred_idx) # 結果のラベル