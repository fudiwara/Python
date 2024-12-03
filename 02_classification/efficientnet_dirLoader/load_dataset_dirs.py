import sys
sys.dont_write_bytecode = True
import pathlib, csv

from PIL import Image
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

import config as cf

class ImageFolder_directory(Dataset):
    def __init__(self, img_dir_path, data_transforms): # 画像フォルダのルートパスを指定
        IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"]
        self.img_paths = sorted([p for p in img_dir_path.glob("**/*") if p.suffix in IMG_EXTS])
        cls_num = []
        for i in range(len(self.img_paths)): # 画像のパス一覧の一つ親側のフォルダ名からクラスIDを得る
            cls_num.append(int(self.img_paths[i].parent.name))
            # print(cls_num[i], str(self.img_paths[i]))
        self.cls_num = cls_num
        self.transforms = data_transforms

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = img.resize((cf.cellSize, cf.cellSize))
        
        img = self.transforms(img)
        return img, self.cls_num[idx]

    def __len__(self): # ディレクトリ内の画像ファイルの数
        return len(self.img_paths)

if __name__ == "__main__":
    import time
    f_tm = time.time()

    dataset_raw = ImageFolder_directory(pathlib.Path(sys.argv[1]), cf.data_transforms)

    dataloader = DataLoader(dataset_raw, batch_size = 5) # , shuffle = True)

    for n, (data, label) in enumerate(dataloader):
        # print(labels[n], imgpaths[n])
        print(n)
        print(data.shape)
        print(label)
        print(label.shape)

        torchvision.utils.save_image(data, f"_{n:03}_i.png", value_range=(-1.0,1.0), normalize=True)

        if n == 5: break

    print()
    print(time.time() - f_tm)
