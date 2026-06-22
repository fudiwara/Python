import sys
import pathlib
import numpy as np
from PIL import Image
import cv2 as cv

import torch
from torch import nn

import config as cf

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

model_path = sys.argv[1]
image_path = sys.argv[2]
file_name = pathlib.Path(image_path)

# model
model = cf.GeneratorAE().to(DEVICE)
model = nn.DataParallel(model)

if DEVICE == "cuda":
    state = torch.load(model_path)
else:
    state = torch.load(model_path, map_location=torch.device("cpu"))
model.load_state_dict(state)
model.eval()

# ===== 前処理を学習時に合わせる =====
# 学習時: RGB -> (to tensor) -> Gray化して[-1,1]
# ここではRGB読込後、OpenCVのRGB2GRAYでL(輝度)を作る
img_rgb = Image.open(image_path).convert("RGB")
i_w, i_h = img_rgb.size
img_rgb = img_rgb.resize((cf.cellSize, cf.cellSize), resample=Image.BICUBIC)

rgb_np = np.array(img_rgb, dtype=np.uint8)                  # H,W,3 (RGB)
gray_np = cv.cvtColor(rgb_np, cv.COLOR_RGB2GRAY)            # H,W
gray_f = gray_np.astype(np.float32) / 255.0                 # [0,1]
gray_n = gray_f * 2.0 - 1.0                                 # [-1,1]

data = torch.from_numpy(gray_n).unsqueeze(0).unsqueeze(0).to(DEVICE)  # 1,1,H,W

with torch.no_grad():
    output = model(data)  # [-1,1], 1,3,H,W

tmp = output[0].permute(1, 2, 0).detach().cpu().numpy()     # H,W,3
tmp = ((tmp + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)

img_bgr = cv.cvtColor(tmp, cv.COLOR_RGB2BGR)
img_out = cv.resize(img_bgr, (i_w, i_h), interpolation=cv.INTER_LANCZOS4)

output_filename = file_name.stem + "_ae_gc.png"
cv.imwrite(output_filename, img_out)
print("saved:", output_filename)