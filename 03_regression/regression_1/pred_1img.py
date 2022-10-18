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
model = cf.build_model().to(DEVICE)
if DEVICE == "cuda":  model.load_state_dict(torch.load(model_path))
else: model.load_state_dict(torch.load(model_path, torch.device("cpu")))
model.eval()

# 画像の読み込み・変換
img = Image.open(image_path).convert('RGB')
data_transforms = T.Compose([T.Resize(cf.cellSize), T.ToTensor()])
data = data_transforms(img)
data = data.unsqueeze(0) # テンソルに変換してから1次元追加
# print(data)
# print(data.shape)

# 推定処理
data = data.to(DEVICE)
outputs = model(data)
# print(outputs)

# 結果には正規化用の係数を乗算する
pred_val = outputs[0].item() * cf.val_rate
print(pred_val)