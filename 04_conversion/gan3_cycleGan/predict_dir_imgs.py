# python pred_dir_img.py モデルファイル名 画像ディレクトリ名 出力先のパス
import sys, pathlib
sys.dont_write_bytecode = True
import numpy as np
from PIL import Image
import cv2 as cv

import torch
from torch import nn
import torchvision.transforms as T

import config as cf

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
model_path = sys.argv[1] # モデルのパス
image_dir_path = sys.argv[2] # 入力画像が入っているディレクトリのパス
output_path = pathlib.Path(sys.argv[3]) # 出力先のパス
if(not output_path.exists()): output_path.mkdir() # 出力先がない場合はフォルダ生成

# モデルの定義と読み込みおよび評価用のモードにセットする
model = cf.Generator(3, cf.resBlocks).to(DEVICE)
if DEVICE == "cuda": model.load_state_dict(torch.load(model_path))
else: model.load_state_dict(torch.load(model_path, torch.device("cpu")))
model.eval()
data_transforms = T.Compose([T.Resize(cf.cellSize), T.ToTensor()])

exts = [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"] # 処理対象の拡張子
fileList = list(pathlib.Path(image_dir_path).iterdir())
fileList.sort()
for i in range(len(fileList)):
    if fileList[i].is_file() and (fileList[i].suffix in exts): # ファイルのみ処理する
        file_name = fileList[i]

        img = Image.open(file_name).convert("RGB") # カラー指定で開く
        img_src = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR) # 後の処理用にnpメモリも用意する
        i_w, i_h = img.size
        data = data_transforms(img).unsqueeze(0) # テンソルに変換してから1次元追加
        # print(data)
        # print(data.shape)

        data = data.to(DEVICE)
        with torch.no_grad(): # 推定のために勾配計算の無効化モードで
            output = model(data) # 推定処理
        tmp = output[0,:,:,:].permute(1, 2, 0) # 画像出力用に次元の入れ替え
        tmp = tmp.to("cpu").detach().numpy() # np配列に変換
        img_tmp = (tmp * 255).astype(np.uint8) # 0-1の範囲なので255倍して画像用データへ
        img_dst = cv.cvtColor(img_tmp, cv.COLOR_RGB2BGR)
        img_ssize_dst = cv.resize(img_dst, (i_w, i_h), interpolation = cv.INTER_LANCZOS4)

        output_filename = file_name.stem + "_cg.jpg"
        cv.imwrite(str(output_path / output_filename), img_ssize_dst) 
