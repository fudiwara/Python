import sys, os, random
sys.dont_write_bytecode = True
import pathlib
import numpy as np
from PIL import Image, ImageDraw

import torch
from torch.utils.data import Dataset, DataLoader

import pyt_det.transforms as T

def collate_fn(batch):
    return tuple(zip(*batch))

def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor()) # PIL imageをPyTorch Tensorに変換
    transforms.append(T.ConvertImageDtype(torch.float))
    if train: transforms.append(T.RandomHorizontalFlip(0.5)) # 訓練中はランダムで水平に反転
    return T.Compose(transforms)

def yolo_to_xyxy(xc, yc, w, h, img_w, img_h):
    # xc, yc, w, h は正規化値(0..1)
    x_center, y_center = xc * img_w, yc * img_h
    box_w, box_h = w * img_w, h * img_h
    x0 = int(round(x_center - box_w / 2))
    y0 = int(round(y_center - box_h / 2))
    x1 = int(round(x_center + box_w / 2))
    y1 = int(round(y_center + box_h / 2))
    # 画像範囲にクリップ
    return max(0, min(img_w - 1, x0)), max(0, min(img_h - 1, y0)), max(0, min(img_w - 1, x1)), max(0, min(img_h - 1, y1))

class annotation_yolotxt_det(Dataset):
    def __init__(self, dataset_path, transforms): # アノテーションファイルへのパスを指定
        IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif", ".webp"] # 画像拡張子
        self.img_paths = sorted([p for p in dataset_path.iterdir() if p.suffix.lower() in IMG_EXTS])
        self.transforms = transforms

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size

        yolo_txt_path = img_path.with_suffix(".txt") # 画像ファイル名と対のTXTファイル
        with open(yolo_txt_path, "r", encoding = "utf-8") as f: # ラベルと座標の読み込み
            lines = [line.strip() for line in f if line.strip()]

        # それぞれを配列に格納する
        boxes = []
        labels = []
        areas = []
        iscrowd = []

        if len(lines) == 0: # アノテーションがなければ空のターゲットを作成
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:

            for line in lines: # 行ごとに矩形を描画
                parts = line.split()
                if len(parts) != 5:
                    continue # 不正行はスキップ

                cls_id = int(parts[0]) + 1 # YOLO形式だと0スタートなので1プラス
                xc = float(parts[1])
                yc = float(parts[2])
                w  = float(parts[3])
                h  = float(parts[4])

                # 範囲外の正規化値はクリップ
                xc = max(0.0, min(1.0, xc))
                yc = max(0.0, min(1.0, yc))
                w  = max(0.0, min(1.0, w))
                h  = max(0.0, min(1.0, h))

                x0, y0, x1, y1 = yolo_to_xyxy(xc, yc, w, h, img_w, img_h)
                if x1 <= x0 or y1 <= y0:
                    continue
                area = (x1 - x0) * (y1 - y0)
                boxes.append([x0, y0, x1, y1])
                labels.append(cls_id)
                areas.append(area)
                iscrowd.append(0)
                # print(idx, i, cls_id, x0, y0, x1, y1)
        
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
    data_path = pathlib.Path(sys.argv[1])
    train_dataset = annotation_yolotxt_det(data_path, get_transform(train=True))
    val_dataset = annotation_yolotxt_det(data_path, get_transform(train=False))

    indices = torch.randperm(len(train_dataset)).tolist()
    train_data_size = int(0.8 * len(indices))
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
