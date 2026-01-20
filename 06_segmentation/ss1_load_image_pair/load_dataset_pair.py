import sys, os, random
sys.dont_write_bytecode = True
import pathlib

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

import config as cf
import pyt_det.transforms as T

def collate_fn(batch):
    return tuple(zip(*batch))

def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor()) # PIL imageをPyTorch Tensorに変換
    transforms.append(T.ConvertImageDtype(torch.float))
    if train: transforms.append(T.RandomHorizontalFlip(0.5)) # 訓練中はランダムで水平に反転
    return T.Compose(transforms)

class ImageFolder_pair(Dataset):
    def __init__(self, img_dir_path, ant_dir_path, transforms): # 画像とアノテーションPNGへのパス
        self.img_dir_path, self.ant_dir_path = get_images_list(img_dir_path, ant_dir_path)
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.img_dir_path)

    def __getitem__(self, idx):
        img_path = self.img_dir_path[idx]
        mask_path = self.ant_dir_path[idx]
        # print(img_path, mask_path)
        img = Image.open(img_path).convert("RGB") # 画像はそのままひらく
        mask = Image.open(mask_path) # maskはconvertしない

        mask = np.array(mask)
        obj_ids = np.unique(mask) # 各インスタンス・クラスの値を取得
        obj_ids = obj_ids[1:] # 0は背景なので削除
        # print(obj_ids)

        masks = mask == obj_ids[:, None, None] # カラー・エンコードされたマスクを、True/Falseで

        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs): # 各マスクのバウンディングボックス
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64) # クラス数が増えたら要修正
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64) # すべてのインスタンスを iscrowd=0 と仮定

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        img, target = self.transforms(img, target)

        return img, target

def get_images_list(dirA, dirB): # 2つのディレクトリを比較して同じファイル名があるパスだけをお互いに残す
    exts = [".jpg", ".png", ".jpeg"]
    setA, setB = set(), set()

    fileList = list(pathlib.Path(dirA).glob("**/*"))
    fileList.sort()
    for i in range(len(fileList)): # ターゲットのディレクトリ内を順にチェックしていく
        if fileList[i].is_file() and (fileList[i].suffix.lower() in exts): # 指定拡張子のみ処理する
            setA.add(fileList[i].name) # ファイル名のみをsetに追加 (stemは使わない)

    fileList = list(pathlib.Path(dirB).glob("**/*"))
    fileList.sort()
    for i in range(len(fileList)): # ターゲットのディレクトリ内を順にチェックしていく
        if fileList[i].is_file() and (fileList[i].suffix.lower() in exts): # 指定拡張子のみ処理する
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

    img_dir_path = sys.argv[1]
    annot_dir_name = sys.argv[2]
    train_dataset = ImageFolder_pair(img_dir_path, annot_dir_name, get_transform(train=True))
    val_dataset = ImageFolder_pair(img_dir_path, annot_dir_name, get_transform(train=False))
    print(train_dataset)

    indices = torch.randperm(len(train_dataset)).tolist()
    train_data_size = int(cf.splitRateTrain * len(indices))
    train_dataset = torch.utils.data.Subset(train_dataset, indices[:train_data_size])
    val_dataset = torch.utils.data.Subset(val_dataset, indices[train_data_size:])

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)

    print("don pre-pro")
    for n, (imgs, lbls) in enumerate(train_loader):
    #     # print(labels[n], imgpaths[n])
        print("itr: ", n)
        print(imgs)
        # print(label.shape)
        print(lbls)
        # if n == 0: break

    """
    for n, (imgs, lbls) in enumerate(val_loader):
    #     # print(labels[n], imgpaths[n])
        print("itr: ", n)
        print(imgs)
        # print(label.shape)
        print(lbls)
        # if n == 0: break
    """
