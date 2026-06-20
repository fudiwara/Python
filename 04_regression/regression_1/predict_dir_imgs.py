import sys, pathlib
sys.dont_write_bytecode = True
import torch
import config as cf
import cv2 as cv

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
model_path = sys.argv[1] # モデルのパス
image_dir_path = pathlib.Path(sys.argv[2]) # 入力画像が入っているディレクトリのパス

# モデルの定義と読み込みおよび評価用のモードにセットする
model = cf.build_model("eval").to(DEVICE)
model.load_state_dict(torch.load(model_path, map_location = DEVICE, weights_only = False))
model.eval()

img_paths = sorted([p for p in image_dir_path.iterdir() if p.suffix.lower() in cf.img_ext])
for i in range(len(img_paths)):
    image_path = img_paths[i]

    # 画像の読み込み・変換
    img = cv.cvtColor(cv.imread(image_path), cv.COLOR_BGR2RGB)
    data = cf.transforms_eval(img).unsqueeze(0).to(DEVICE) # テンソルに変換してから1次元追加

    with torch.no_grad(): # 推定のために勾配計算の無効化モードで
        outputs = model(data) # 推定処理

    # 結果には正規化用の係数を乗算する
    pred_val = outputs[0].item() * cf.val_rate
    print(pred_val, image_path.name)
