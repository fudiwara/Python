import sys, os
sys.dont_write_bytecode = True
import pathlib
from PIL import Image

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

import config as cf

class ImageFolder_p2p2(Dataset):
    def __init__(self, imgs_dir0, imgs_dir1): # 画像ファイルのパス一覧
        self.img_paths0, self.img_paths1 = get_images_list(imgs_dir0, imgs_dir1)
        self.transform = data_transforms

    def __getitem__(self, idx):
        # 画像読み込み
        path0 = self.img_paths0[idx]
        path1 = self.img_paths1[idx]
        img0 = Image.open(path0).convert("RGB")
        img1 = Image.open(path1).convert("RGB")
        img0 = self.transform(img0)
        img1 = self.transform(img1)
        return img0, img1

    def __len__(self): # ディレクトリ内の画像ファイルの数
        return len(self.img_paths0)

# データ変換
data_transforms = T.Compose([
    T.Resize(int(cf.cellSize * 1.2)),
    # T.RandomRotation(degrees = 15),
    # T.RandomApply([T.GaussianBlur(5, sigma = (0.1, 5.0))], p = 0.5),
    # T.ColorJitter(brightness = 0, contrast = 0, saturation = 0, hue = [-0.2, 0.2]),
    # T.RandomHorizontalFlip(0.5),
    T.CenterCrop(cf.cellSize),
    T.ToTensor()
    ])

def load_datasets(imgs_dir0, imgs_dir1):
    datasets_raw = ImageFolder_p2p2(imgs_dir0, imgs_dir1)
    cf.dataset_size = len(datasets_raw)
    train_loader = DataLoader(datasets_raw, batch_size=cf.batchSize, shuffle=True, num_workers = os.cpu_count(), pin_memory=True)
    return train_loader

def get_images_list(dirA, dirB): # 2つのディレクトリを比較して同じファイル名があるパスだけをお互いに残す
    exts = [".jpg", ".png", ".jpeg"]
    setA, setB = set(), set()

    fileList = list(pathlib.Path(dirA).glob("**/*"))
    fileList.sort()
    for i in range(len(fileList)): # ターゲットのディレクトリ内を順にチェックしていく
        if fileList[i].is_file() and (fileList[i].suffix.lower()  in exts): # 指定の拡張子のみ処理する
            setA.add(fileList[i].name) # ファイル名のみをsetに追加 (stemは使わない)

    fileList = list(pathlib.Path(dirB).glob("**/*"))
    fileList.sort()
    for i in range(len(fileList)): # ターゲットのディレクトリ内を順にチェックしていく
        if fileList[i].is_file() and (fileList[i].suffix.lower()  in exts): # 指定の拡張子のみ処理する
            setB.add(fileList[i].name) # ファイル名のみをsetに追加 (stemは使わない)
    
    andSet = setA & setB # ファイル名のみで積集合を得る
    andList = list(andSet)
    andList.sort() # 順番でソート
    # print(andList)

    listA, listB = [], []
    pathA, pathB = pathlib.Path(dirA).resolve(), pathlib.Path(dirB).resolve()
    for i in range(len(andList)):
        listA.append(pathA / andList[i]) # フルパスでどちらにもあるリストを生成する
        listB.append(pathB / andList[i])

    return listA, listB

if __name__ == "__main__":
    import time
    f_tm = time.time()

    dataloader = load_datasets(sys.argv[1], sys.argv[2])

    for n, (img0, img1) in enumerate(dataloader):
        # print(labels[n], imgpaths[n])
        print(n)
        print(img0.shape)
        print(img1.shape)
        if n == 2: break

    print()
    print(time.time() - f_tm)
