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
image_dir_path = pathlib.Path(sys.argv[2])

model = cf.GeneratorAE().to(DEVICE)
# DataParallel学習済みモデルを読むため
model = nn.DataParallel(model)

if DEVICE == "cuda":
    model.load_state_dict(torch.load(model_path))
else:
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

tf = T.Compose([
    T.Resize(cf.cellSize),
    T.ToTensor(),
    T.Normalize(mean=(0.5,), std=(0.5,))
])

IMG_EXTS = [".jpg", ".png", ".jpeg", ".bmp"]
file_list = sorted([p for p in image_dir_path.iterdir() if p.suffix.lower() in IMG_EXTS])

for file_name in file_list:
    img = Image.open(file_name).convert("L")
    i_w, i_h = img.size
    data = tf(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(data)

    tmp = output[0].permute(1, 2, 0).to("cpu").numpy()
    tmp = ((tmp + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
    img_dst = cv.cvtColor(tmp, cv.COLOR_RGB2BGR)
    img_ssize_dst = cv.resize(img_dst, (i_w, i_h), interpolation=cv.INTER_LANCZOS4)

    output_filename = file_name.stem + "_ae_gc.png"
    cv.imwrite(str(image_dir_path / output_filename), img_ssize_dst)
    print("saved:", output_filename)