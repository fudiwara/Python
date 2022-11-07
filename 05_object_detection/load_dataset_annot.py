import sys, os, random
sys.dont_write_bytecode = True
import pathlib
import numpy as np
from PIL import Image, ImageDraw

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

class ImageFolderAnnotationRect(Dataset):
    def __init__(self, img_dir_path, annotations_path, transforms): # アノテーションファイルへのパスを指定
        lines = []
        with open(annotations_path, "r") as f:# self.filelines = f.read().split('\n')
            for line in f:
                if 1 < len(line.split(" ")):
                    lines.append(line)
        self.filelines = lines
        # print(self.filelines)
        self.img_dir_path = pathlib.Path(img_dir_path)
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.filelines)

    def __getitem__(self, idx):
        l = self.filelines[idx].split(" ") # スペース区切りのセットリスト
        imgname = l[0]
        img_path = self.img_dir_path / imgname
        # print(img_path, l)
        img = Image.open(img_path).convert("RGB")

        # それぞれを配列に格納する
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        for i in range(len(l) - 1):
            p = l[i + 1].split(",") # カンマ区切りのセットリスト
            x0 = float(p[0]) # 実数で読み込む
            y0 = float(p[1])
            x1 = float(p[2])
            y1 = float(p[3])
            cls = int(p[4])
            area = (x1 - x0) * (y1 - y0)
            boxes.append([x0, y0, x1, y1])
            labels.append(cls)
            areas.append(area)
            iscrowd.append(0)
            # print(idx, i, cls, x0, y0, x1, y1)
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        areas = torch.as_tensor(areas)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = areas
        target["iscrowd"] = iscrowd

        img, target = self.transforms(img, target)

        return img, target

if __name__ == "__main__":

    img_dir_path = sys.argv[1]
    annot_file_name = sys.argv[2]
    train_dataset = ImageFolderAnnotationRect(img_dir_path, annot_file_name, get_transform(train=True))
    val_dataset = ImageFolderAnnotationRect(img_dir_path, annot_file_name, get_transform(train=False))

    indices = torch.randperm(len(train_dataset)).tolist()
    train_data_size = int(cf.splitRateTrain * len(indices))
    train_dataset = torch.utils.data.Subset(train_dataset, indices[:train_data_size])
    val_dataset = torch.utils.data.Subset(val_dataset, indices[train_data_size:])

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)

    for n, (imgs, lbls) in enumerate(train_loader):
    #     # print(labels[n], imgpaths[n])
        print("itr: ", n)
        print(imgs)
        # print(label.shape)
        print(lbls)
        # if n == 0: break

    for n, (imgs, lbls) in enumerate(val_loader):
    #     # print(labels[n], imgpaths[n])
        print("itr: ", n)
        print(imgs)
        # print(label.shape)
        print(lbls)
        # if n == 0: break
