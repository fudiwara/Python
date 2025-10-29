import sys, time, os
sys.dont_write_bytecode = True
import numpy as np
from PIL import Image
import cv2 as cv
import pathlib

import torch
from torch import nn
import torchvision.transforms as T

import config as cf

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
torch.backends.cudnn.benchmark = True
model_path = sys.argv[1] # モデルのパス
image_path = sys.argv[2] # 入力画像のパス
file_name = pathlib.Path(sys.argv[2])

model = cf.Generator().to(DEVICE)
model = nn.DataParallel(model) # 学習時とあわせてDataParallelを指定する

if DEVICE == "cuda":  model.load_state_dict(torch.load(model_path))
else: model.load_state_dict(torch.load(model_path, torch.device("cpu")))
model.eval()

img = Image.open(image_path).convert("RGB") # カラー指定で開く
i_w, i_h = img.size
img = img.resize((cf.cellSize, cf.cellSize))
data_transforms = T.Compose([T.Resize(cf.cellSize), T.ToTensor()])
data = data_transforms(img).unsqueeze(0) # テンソルに変換してから1次元追加
print(data)
print(data.shape)

data = data.to(DEVICE)
with torch.no_grad(): # 推定のために勾配計算の無効化モードで
    output = model(data) # 推定処理
print(output)
print(output.shape)

tmp = output[0,:,:,:].permute(1, 2, 0) # 画像出力用に次元の入れ替え
tmp = tmp.to("cpu").detach().numpy() # np配列に変換
img_tmp = (tmp*255).astype(np.uint8) # 0-1の範囲なので255倍して画像用データへ
img_dst = cv.cvtColor(img_tmp, cv.COLOR_RGB2BGR)
img_ssize_dst = cv.resize(img_dst, (i_w, i_h), interpolation = cv.INTER_LANCZOS4)
outputFIlename = file_name.stem + "_p2p.png"
cv.imwrite(outputFIlename, img_ssize_dst) 
