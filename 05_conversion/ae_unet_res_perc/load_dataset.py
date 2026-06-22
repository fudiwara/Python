import sys, os
sys.dont_write_bytecode = True
import pathlib
from PIL import Image

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

import config as cf


class ImageFolderAE(Dataset):
    IMG_EXT = [".jpg", ".jpeg", ".png", ".bmp"]

    def __init__(self, imgs_dir):
        self.img_paths = self._get_img_paths(imgs_dir)
        self.transform = data_transforms

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = Image.open(path).convert("RGB")
        color, gray = self.transform(img)  # color:3ch, gray:1ch
        return color, gray

    def _get_img_paths(self, imgs_dir):
        imgs_dir = pathlib.Path(imgs_dir)
        img_paths = [p for p in imgs_dir.iterdir() if p.suffix.lower() in ImageFolderAE.IMG_EXT]
        return sorted(img_paths)

    def __len__(self):
        return len(self.img_paths)


class ColorAndGray(object):
    def __call__(self, img):
        gray = img.convert("L")
        return img, gray


class MultiInputWrapper(object):
    def __init__(self, base_func):
        self.base_func = base_func

    def __call__(self, xs):
        if isinstance(self.base_func, list):
            return [f(x) for f, x in zip(self.base_func, xs)]
        return [self.base_func(x) for x in xs]


data_transforms = T.Compose([
    T.Resize(int(cf.cellSize * 1.2)),
    T.RandomRotation(degrees=15),
    T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 5.0))], p=0.5),
    T.RandomHorizontalFlip(0.5),
    T.CenterCrop(cf.cellSize),
    ColorAndGray(),
    MultiInputWrapper(T.ToTensor()),
    MultiInputWrapper([
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # color -> [-1,1]
        T.Normalize(mean=(0.5,), std=(0.5,))                     # gray  -> [-1,1]
    ])
])


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