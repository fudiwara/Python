import sys, os
sys.dont_write_bytecode = True
import pathlib
from PIL import Image

import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

import config as cf

def load_datasets(imgs_dir):
    datasets_raw = ImageFolder_p2p1(imgs_dir)
    cf.dataset_size = len(datasets_raw)

    nw = min(4, os.cpu_count() if os.cpu_count() is not None else 2)

    train_loader = DataLoader(datasets_raw, batch_size=cf.batchSize, shuffle=True, num_workers=nw, pin_memory=True, drop_last=False)
    return train_loader

class ImageFolder_p2p1(Dataset):
    IMG_EXT = [".jpg", ".jpeg", ".png", ".bmp"]

    def __init__(self, imgs_dir):
        self.img_paths = self._get_img_paths(imgs_dir)
        self.transform = data_transforms

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = Image.open(path).convert("RGB")
        color, gray = self.transform(img)
        return color, gray

    def _get_img_paths(self, imgs_dir):
        data = sorted([p for p in imgs_dir.iterdir() if p.suffix.lower() in self.IMG_EXT])
        return data[:cf.dataset_size]

    def __len__(self):
        return len(self.img_paths)

class ColorAndGray:
    def __call__(self, img):
        gray = img.convert("L")
        return img, gray

class MultiInputWrapper:
    def __init__(self, base_func):
        self.base_func = base_func

    def __call__(self, xs):
        if isinstance(self.base_func, list):
            return [f(x) for f, x in zip(self.base_func, xs)]
        return [self.base_func(x) for x in xs]

data_transforms = T.Compose([
    T.Resize(int(cf.cellSize * 1.1)),
    T.RandomHorizontalFlip(0.5),
    T.CenterCrop(cf.cellSize),
    ColorAndGray(),
    MultiInputWrapper(T.ToTensor()),
    MultiInputWrapper([
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        T.Normalize(mean=(0.5,), std=(0.5,))
    ])
])



