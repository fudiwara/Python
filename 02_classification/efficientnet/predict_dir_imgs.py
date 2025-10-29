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

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"] # 処理対象の拡張子
img_paths = sorted([p for p in image_dir_path.iterdir() if p.suffix in cf.IMG_EXTS])
for i in range(len(fileList)):
    image_path = img_paths[i]
    img = Image.open(image_path).convert("RGB") # カラー指定で開く
    data = data_transforms(img).unsqueeze(0) # テンソルに変換してから1次元追加
    # print(data)
    # print(data.shape)

    # 推定処理
    data = data.to(DEVICE)

    with torch.no_grad(): # 推定のために勾配計算の無効化モードで
        outputs = model(data)
    _, preds = torch.max(outputs, 1) # 1次元目の中の最大値を得る(最大値と最大値のインデックス)
    pred_idx = preds.cpu().numpy().tolist() # tensorから数値へ
    
    np.set_printoptions(precision = 3)
    pred_val = outputs[0].to("cpu").detach().numpy().copy() # 各クラスの推定値

    print(image_path.name) # ファイル名
    print(pred_val) # 推定結果の表示
    print(pred_idx[0]) # 結果のラベル