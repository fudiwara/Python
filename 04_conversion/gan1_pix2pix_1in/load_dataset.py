import sys, os
sys.dont_write_bytecode = True
import pathlib
from PIL import Image

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

import config as cf

class ImageFolder_p2p1(Dataset):
    IMG_EXT = [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"]

    def __init__(self, imgs_dir): # 画像ファイルのパス一覧
        self.img_paths = self._get_img_paths(imgs_dir)
        self.transform = data_transforms
        self.ftf = T.Compose([T.ToTensor()])

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = Image.open(path).convert("RGB") # 画像読み込み
        img = self.transform(img)
        return img

    def _get_img_paths(self, imgs_dir): # 指定ディレクトリ内の画像ファイルパス一覧
        imgs_dir = pathlib.Path(imgs_dir)
        img_paths = [p for p in imgs_dir.iterdir() if p.suffix in ImageFolder_p2p1.IMG_EXT]
        return img_paths

    def __len__(self): # ディレクトリ内の画像ファイルの数
        return len(self.img_paths)

class ColorAndGray(object):
    def __call__(self, img):
        gray = img.convert("L") # ToTensor()の前に呼ぶ場合はimgはPIL
        return img, gray

class MultiInputWrapper(object): # 複数の入力をtransformsに展開するラッパ
    def __init__(self, base_func):
        self.base_func = base_func

    def __call__(self, xs):
        if isinstance(self.base_func, list): return [f(x) for f, x in zip(self.base_func, xs)]
        else: return [self.base_func(x) for x in xs]

# データ変換
data_transforms = T.Compose([
    T.Resize(int(cf.cellSize * 1.2)),
    T.RandomRotation(degrees = 15),
    T.RandomApply([T.GaussianBlur(5, sigma = (0.1, 5.0))], p = 0.5),
    # T.ColorJitter(brightness = 0, contrast = 0, saturation = 0, hue = [-0.2, 0.2]),
    T.RandomHorizontalFlip(0.5),
    T.CenterCrop(cf.cellSize),
    ColorAndGray(),
    MultiInputWrapper(T.ToTensor()),
    MultiInputWrapper([T.Normalize(mean=(0.5,0.5,0.5,), std=(0.5,0.5,0.5,)), T.Normalize(mean=(0.5,), std=(0.5,))])
    ])

def load_datasets(imgs_dir):
    datasets_raw = ImageFolder_p2p1(imgs_dir)
    cf.dataset_size = len(datasets_raw)
    train_loader = DataLoader(datasets_raw, batch_size=cf.batchSize, shuffle=True, num_workers = os.cpu_count(), pin_memory=True)
    return train_loader

if __name__ == "__main__":
    import time
    f_tm = time.time()

    transform = T.Compose([T.Resize(256), T.ToTensor()])

    dataset_raw = ImageFolder(sys.argv[1], transform)

    dataloader = DataLoader(dataset_raw, batch_size=3)

    for n, (data, label) in enumerate(dataloader):
        # print(labels[n], imgpaths[n])
        print(n)
        print(data.shape)
        print(label)
        print(label.shape)
        if n == 2: break

    print()
    print(time.time() - f_tm)
