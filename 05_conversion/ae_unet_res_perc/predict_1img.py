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

img = Image.open(image_path).convert("L")
i_w, i_h = img.size
img = img.resize((cf.cellSize, cf.cellSize))

tf = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.5,), std=(0.5,))
])
data = tf(img).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    output = model(data)  # [-1,1]

tmp = output[0].permute(1, 2, 0).to("cpu").numpy()
tmp = ((tmp + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
img_dst = cv.cvtColor(tmp, cv.COLOR_RGB2BGR)
img_ssize_dst = cv.resize(img_dst, (i_w, i_h), interpolation=cv.INTER_LANCZOS4)

output_filename = file_name.stem + "_ae_gc.png"
cv.imwrite(output_filename, img_ssize_dst)
print("saved:", output_filename)