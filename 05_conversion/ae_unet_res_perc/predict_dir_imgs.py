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
image_dir_path = pathlib.Path(sys.argv[2])

# model
model = cf.GeneratorAE().to(DEVICE)
model = nn.DataParallel(model)

if DEVICE == "cuda":
    state = torch.load(model_path)
else:
    state = torch.load(model_path, map_location=torch.device("cpu"))
model.load_state_dict(state)
model.eval()

IMG_EXTS = [".jpg", ".png", ".jpeg", ".bmp"]
file_list = sorted([p for p in image_dir_path.iterdir() if p.suffix.lower() in IMG_EXTS])

for file_name in file_list:
    # ===== 前処理を学習時に合わせる =====
    img_rgb = Image.open(file_name).convert("RGB")
    i_w, i_h = img_rgb.size
    img_rgb = img_rgb.resize((cf.cellSize, cf.cellSize), resample=Image.BICUBIC)

    rgb_np = np.array(img_rgb, dtype=np.uint8)
    gray_np = cv.cvtColor(rgb_np, cv.COLOR_RGB2GRAY)
    gray_f = gray_np.astype(np.float32) / 255.0
    gray_n = gray_f * 2.0 - 1.0

    data = torch.from_numpy(gray_n).unsqueeze(0).unsqueeze(0).to(DEVICE)  # 1,1,H,W
    
    with torch.no_grad():
        pred_ab = model(data)[0].detach().cpu().clamp(-1, 1)   # 2,H,W

    # Lは入力時のgray_nから復元
    L_255 = ((gray_n + 1.0) * 0.5 * 255.0).astype(np.float32)  # H,W
    ab_255 = (pred_ab.permute(1, 2, 0).numpy() * 127.0 + 128.0).astype(np.float32)  # H,W,2

    lab = np.zeros((cf.cellSize, cf.cellSize, 3), dtype=np.float32)
    lab[:, :, 0] = L_255
    lab[:, :, 1:3] = ab_255
    lab = np.clip(lab, 0, 255).astype(np.uint8)

    rgb = cv.cvtColor(lab, cv.COLOR_LAB2RGB)                   # H,W,3
    img_out = cv.resize(rgb, (i_w, i_h), interpolation=cv.INTER_LANCZOS4)

    # OpenCVで保存するのでBGRに変換
    output_filename = file_name.stem + "_ae_gc.png"
    cv.imwrite(str(image_dir_path / output_filename), cv.cvtColor(img_out, cv.COLOR_RGB2BGR))