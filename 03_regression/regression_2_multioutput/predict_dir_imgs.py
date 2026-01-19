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
image_dir_path = sys.argv[2] # 入力画像が入っているディレクトリのパス

# モデルの定義と読み込みおよび評価用のモードにセットする
model = cf.build_model("eval").to(DEVICE)
if DEVICE == "cuda": model.load_state_dict(torch.load(model_path))
else: model.load_state_dict(torch.load(model_path, torch.device("cpu")))
model.eval()
data_transforms = T.Compose([T.Resize(cf.cellSize), T.CenterCrop(cf.cellSize), T.ToTensor()])

IMG_EXTS = [".jpg", ".png", ".jpeg"] # 処理対象の拡張子
fileList = sorted([p for p in image_dir_path.iterdir() if p.suffix.lower() in IMG_EXTS])
for i in range(len(fileList)):
    image_path = fileList[i]

    # 画像の読み込み・変換
    img = Image.open(image_path).convert("RGB")
    data = data_transforms(img).unsqueeze(0) # テンソルに変換してから1次元追加

    # 推定処理
    data = data.to(DEVICE)
    with torch.no_grad(): # 推定のために勾配計算の無効化モードで
        outputs = model(data)

    # 結果には正規化用の係数を乗算する
    pred_val_0 = outputs[0].item() * cf.val_rate_0
    pred_val_1 = outputs[1].item() * cf.val_rate_1
    print(f"{pred_val_0:.3f} {pred_val_1:.3f} {image_path.name}")
