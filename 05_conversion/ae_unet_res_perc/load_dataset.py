import sys, os
sys.dont_write_bytecode = True
import pathlib
import numpy as np
from PIL import Image
import cv2 as cv

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

import config as cf


class ImageFolderAE(Dataset):
    IMG_EXT = [".jpg", ".jpeg", ".png", ".bmp"]

    def __init__(self, imgs_dir):
        self.img_paths = self._get_img_paths(imgs_dir)
        self.resize = T.Resize(int(cf.cellSize * 1.2))
        self.center_crop = T.CenterCrop(cf.cellSize)
        self.hflip = T.RandomHorizontalFlip(0.5)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = Image.open(path).convert("RGB")

        # 幾何変換はRGBのまままとめて
        img = self.resize(img)
        img = self.hflip(img)
        img = self.center_crop(img)

        # PIL -> np(RGB) -> Lab
        rgb = np.array(img, dtype=np.uint8)
        lab = cv.cvtColor(rgb, cv.COLOR_RGB2LAB).astype(np.float32)

        # OpenCV Lab: L[0,255], a,b[0,255]
        L = lab[:, :, 0:1]         # H,W,1
        ab = lab[:, :, 1:3]        # H,W,2

        # 正規化
        L = (L / 255.0) * 2.0 - 1.0          # -> [-1,1]
        ab = (ab - 128.0) / 127.0            # -> おおむね[-1,1]

        # CHW tensor
        L_t = torch.from_numpy(L.transpose(2, 0, 1)).float()    # 1,H,W
        ab_t = torch.from_numpy(ab.transpose(2, 0, 1)).float()  # 2,H,W

        return ab_t, L_t

    def _get_img_paths(self, imgs_dir):
        imgs_dir = pathlib.Path(imgs_dir)
        img_paths = [p for p in imgs_dir.iterdir() if p.suffix.lower() in ImageFolderAE.IMG_EXT]
        return sorted(img_paths)

    def __len__(self):
        return len(self.img_paths)


def load_datasets(imgs_dir):
    dataset_raw = ImageFolderAE(imgs_dir)
    cf.dataset_size = len(dataset_raw)
    train_loader = DataLoader(
        dataset_raw,
        batch_size=cf.batchSize,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True
    )
    return train_loader