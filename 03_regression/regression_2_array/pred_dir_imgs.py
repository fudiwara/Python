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
model = cf.build_model().to(DEVICE)
if DEVICE == "cuda":  model.load_state_dict(torch.load(model_path))
else: model.load_state_dict(torch.load(model_path, torch.device("cpu")))
model.eval()

data_transforms = T.Compose([T.Resize(cf.cellSize), T.CenterCrop(cf.cellSize), T.ToTensor()])
exts = ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG'] # 処理対象の拡張子
fileList = list(pathlib.Path(image_dir_path).iterdir())
fileList.sort()
for i in range(len(fileList)):
    if fileList[i].is_file() and (fileList[i].suffix in exts): # ファイルのみ処理する
        image_path = fileList[i]

        # 画像の読み込み・変換
        img = Image.open(image_path).convert('RGB')
        data = data_transforms(img)
        data = data.unsqueeze(0)
        # print(data)
        # print(data.shape)

        # 推定処理
        data = data.to(DEVICE)
        outputs = model(data)
        # print(outputs)

        # 結果には正規化用の係数を乗算する
        pred_val_0 = outputs[0][0].item() * cf.val_rate_0
        pred_val_1 = outputs[0][1].item() * cf.val_rate_1
        print(f"{pred_val_0:.3f} {pred_val_1:.3f} {image_path.name}")
