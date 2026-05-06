import sys
sys.dont_write_bytecode = True
import pathlib, csv

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2 as cv

import config as cf

class ImageFolder_directory(Dataset):
    def __init__(self, img_dir_path, data_transforms): # 画像フォルダのルートパスを指定
        IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]
        self.img_paths = sorted([p for p in img_dir_path.glob("**/*") if p.suffix.lower() in IMG_EXTS])
        cls_num = []
        class_names = [0] * cf.classesSize
        for i in range(len(self.img_paths)): # 画像のパス一覧の一つ親側のフォルダ名からクラスIDを得る
            parent_dir_name = self.img_paths[i].parent.name
            cls_id = int(parent_dir_name.split("_")[0]) # フォルダ名を _ で分割した先頭をIDにする
            cls_num.append(cls_id) # 最初の要素をクラスIDとして使用
            if parent_dir_name not in class_names:
                class_names[cls_id] = parent_dir_name # クラス名のリストを作る

            # print(cls_num[i], str(self.img_paths[i]))
        self.cls_num = cls_num
        self.transforms = data_transforms
        self.class_to_idx = {class_names[i]: i for i in range(cf.classesSize)} # クラス名とクラスIDの対応辞書

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv.cvtColor(cv.imread(str(img_path)), cv.COLOR_BGR2RGB)

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
