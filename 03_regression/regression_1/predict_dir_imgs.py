import sys, pathlib
sys.dont_write_bytecode = True
import torch
import torchvision.transforms as T
import numpy as np
import config as cf
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
model_path = sys.argv[1] # モデルのパス
image_dir_path = pathlib.Path(sys.argv[2]) # 入力画像が入っているディレクトリのパス

# モデルの定義と読み込みおよび評価用のモードにセットする
model = cf.build_model("eval").to(DEVICE)
if DEVICE == "cuda": model.load_state_dict(torch.load(model_path, weights_only = False))
else: model.load_state_dict(torch.load(model_path, torch.device("cpu")))
model.eval()
data_transforms = T.Compose([T.Resize(cf.cellSize), T.CenterCrop(cf.cellSize), T.ToTensor()])

img_paths = sorted([p for p in image_dir_path.iterdir() if p.suffix in cf.IMG_EXTS])
for i in range(len(fileList)):
    image_path = img_paths[i]

    # 画像の読み込み・変換
    img = Image.open(image_path).convert("RGB")
    data = data_transforms(img)
    data = data.unsqueeze(0) # テンソルに変換してから1次元追加

    # 推定処理
    data = data.to(DEVICE)
    outputs = model(data)

    # 結果には正規化用の係数を乗算する
    pred_val = outputs[0].item() * cf.val_rate
    print(pred_val, image_path.name)
