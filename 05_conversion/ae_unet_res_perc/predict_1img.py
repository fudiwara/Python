import sys, pathlib
import numpy as np
from PIL import Image
import cv2 as cv

import torch
from torch import nn
import torchvision.transforms as T

import config as cf

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

model_path = sys.argv[1]
image_path = sys.argv[2]
file_name = pathlib.Path(image_path)

model = cf.GeneratorAE().to(DEVICE)
model = nn.DataParallel(model)

if DEVICE == "cuda":
    model.load_state_dict(torch.load(model_path))
else:
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

# 入力グレーをLとして使う
img = Image.open(image_path).convert("L")
i_w, i_h = img.size
img = img.resize((cf.cellSize, cf.cellSize))

tf = T.ToTensor()
L = tf(img)  # [0,1], 1ch
L_n = L * 2.0 - 1.0  # [-1,1]
data = L_n.unsqueeze(0).to(DEVICE)

with torch.no_grad():
    pred_ab = model(data).clamp(-1, 1)  # B,2,H,W

# Lab合成
L_255 = (L_n[0].cpu().numpy() + 1.0) * 0.5 * 255.0           # H,W
ab_255 = pred_ab[0].permute(1, 2, 0).cpu().numpy() * 127.0 + 128.0  # H,W,2

lab = np.zeros((cf.cellSize, cf.cellSize, 3), dtype=np.float32)
lab[:, :, 0] = L_255
lab[:, :, 1:3] = ab_255
lab = np.clip(lab, 0, 255).astype(np.uint8)

rgb = cv.cvtColor(lab, cv.COLOR_LAB2RGB)
bgr = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)
out = cv.resize(bgr, (i_w, i_h), interpolation=cv.INTER_LANCZOS4)

output_filename = file_name.stem + "_ae_lab_gc.png"
cv.imwrite(output_filename, out)
print("saved:", output_filename)