# python pred_1img.py モデルファイル名 画像ファイル名
import sys
sys.dont_write_bytecode = True
import numpy as np
from PIL import Image
import cv2
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

# モデルの定義と読み込みおよび評価用のモードにセットする
model = cf.Generator(3, cf.resBlocks).to(DEVICE)
# model = nn.DataParallel(model) # 学習時とあわせてDataParallelを指定する
if DEVICE == "cuda": model.load_state_dict(torch.load(model_path))
else: model.load_state_dict(torch.load(model_path, torch.device("cpu")))
model.eval()

img = Image.open(image_path).convert("RGB") # カラー指定で開く
img_src = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) # 後の処理用にnpメモリも用意する
i_w, i_h = img.size
data_transforms = T.Compose([T.Resize(cf.cellSize), T.ToTensor()])
data = data_transforms(img)
data = data.unsqueeze(0) # テンソルに変換してから1次元追加
# print(data)
# print(data.shape)

data = data.to(DEVICE)
output = model(data) # 推定処理
tmp = output[0,:,:,:].permute(1, 2, 0) # 画像出力用に次元の入れ替え
tmp = tmp.to("cpu").detach().numpy() # np配列に変換
img_tmp = (tmp * 255).astype(np.uint8) # 0-1の範囲なので255倍して画像用データへ
img_dst = cv2.cvtColor(img_tmp, cv2.COLOR_RGB2BGR)
img_ssize_dst = cv2.resize(img_dst, (i_w, i_h), interpolation = cv2.INTER_LANCZOS4)

cv2.imwrite(file_name.stem + "_cg.jpg", img_ssize_dst) 

# 厚塗り成分 B - A の計算
thick_coat = np.array(img_ssize_dst, dtype = np.float32) - np.array(img_src, dtype = np.float32)
tc_out = np.clip(thick_coat, 0, 255)
tc_out = tc_out.astype(np.uint8)
print(np.count_nonzero(tc_out < 0)) # B - A < 0 となる面積のカウント
print(np.mean(tc_out)) # 厚塗り成分の平均値
tc_gray = cv2.cvtColor(tc_out, cv2.COLOR_BGR2GRAY) # 厚塗り成分のグレー化

cv2.imwrite(file_name.stem + "_tc_0prd.jpg", tc_out) 
cv2.imwrite(file_name.stem + "_tc_1gray.jpg", tc_gray) 

tc_val = cv2.cvtColor(tc_gray, cv2.COLOR_GRAY2BGR) # 合成するためにRGBプレーンに
add_tc = np.array(img_src, dtype = np.float32) + np.array(tc_val, dtype = np.float32) * 0.3333
add_tc = np.clip(add_tc, 0, 255)
img_add_tc = add_tc.astype(np.uint8)

cv2.imwrite(file_name.stem + "_tc_2clc.jpg", img_add_tc) 

tc_mean = np.zeros((i_h, i_w, 3), np.uint8)
tc_mean[::] = int(np.mean(tc_out))
img_add_mean_tcval = np.array(img_src, dtype = np.float32) + np.array(tc_mean, dtype = np.float32)
img_add_mean_tcval = np.clip(img_add_mean_tcval, 0, 255)
img_add_mean_tcval = img_add_mean_tcval.astype(np.uint8)
cv2.imwrite(file_name.stem + "_tc_3mval.jpg", img_add_mean_tcval) 
