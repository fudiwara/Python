import sys
sys.dont_write_bytecode = True
import numpy as np
from PIL import Image
import cv2 as cv
import pathlib

import torch
import torchvision.transforms as T

import config as cf

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

model_path = sys.argv[1]
image_path = sys.argv[2]
file_name = pathlib.Path(image_path)

model = cf.Generator().to(DEVICE)
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()

img = Image.open(image_path).convert("L")
i_w, i_h = img.size
img = img.resize((cf.cellSize, cf.cellSize))

tf = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
x = tf(img).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    y = model(x)

# [-1,1] -> [0,255]
tmp = y[0].permute(1, 2, 0).detach().cpu().numpy()
tmp = np.clip((tmp + 1.0) * 127.5, 0, 255).astype(np.uint8)

img_dst = cv.cvtColor(tmp, cv.COLOR_RGB2BGR)
img_dst = cv.resize(img_dst, (i_w, i_h), interpolation=cv.INTER_CUBIC)

out_name = file_name.stem + "_p2p_gc.png"
cv.imwrite(out_name, img_dst)
print("saved:", out_name)