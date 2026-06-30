import sys, os
sys.dont_write_bytecode = True
import pathlib
from PIL import Image

import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

import config as cf

def load_datasets(imgs_dir0, imgs_dir1):
    datasets_raw = ImageFolder_p2p2(imgs_dir0, imgs_dir1)
    cf.dataset_size = len(datasets_raw)

    nw = min(4, os.cpu_count() if os.cpu_count() is not None else 2)
    train_loader = DataLoader(datasets_raw, batch_size=cf.batchSize, shuffle=True, num_workers=nw, pin_memory=True, drop_last=False)
    return train_loader

class ImageFolder_p2p2(Dataset):
    def __init__(self, imgs_dir0, imgs_dir1):
        self.img_paths0, self.img_paths1 = get_images_list(imgs_dir0, imgs_dir1)
        self.transform = data_transforms

    def __getitem__(self, idx):
        path0 = self.img_paths0[idx]
        path1 = self.img_paths1[idx]
        img0 = Image.open(path0).convert("L")
        img1 = Image.open(path1).convert("RGB")
        img0 = self.transform(img0)
        img1 = self.transform(img1)
        return img1, img0

    def _get_img_paths(self, imgs_dir):
        data = sorted([p for p in imgs_dir.iterdir() if p.suffix.lower() in self.IMG_EXT])
        return data[:cf.dataset_size]

    def __len__(self):
        return len(self.img_paths0)

def get_images_list(dirA, dirB): # 2つのディレクトリを比較して同じファイル名があるパスだけをお互いに残す
    IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]
    setA = set([p.name for p in dirA.iterdir() if p.suffix.lower() in IMG_EXTS])
    setB = set([p.name for p in dirB.iterdir() if p.suffix.lower() in IMG_EXTS])
    andList = sorted(list(setA & setB)) # ファイル名のみで積集合を得る
    
    andList = andList[:cf.dataset_size] # 指定したデータセットのサイズに変更する
    pathA, pathB = pathlib.Path(dirA).resolve(), pathlib.Path(dirB).resolve()
    listA = [pathA / andList[i] for i in range(len(andList))] # フルパスでどちらにもあるリストを生成する
    listB = [pathB / andList[i] for i in range(len(andList))]
    return listA, listB

data_transforms = T.Compose([
    T.Resize(int(cf.cellSize * 1.1)),
    T.CenterCrop(cf.cellSize),
    T.ToTensor()
])

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

# data_transforms = T.Compose([ # color to gray
#     T.Resize(int(cf.cellSize * 1.1)),
#     T.RandomHorizontalFlip(0.5),
#     T.CenterCrop(cf.cellSize),
#     ColorAndGray(),
#     MultiInputWrapper(T.ToTensor()),
#     MultiInputWrapper([
#         T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
#         T.Normalize(mean=(0.5,), std=(0.5,))
#     ])
# ])
