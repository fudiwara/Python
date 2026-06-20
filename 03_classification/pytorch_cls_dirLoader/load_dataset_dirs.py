import sys
sys.dont_write_bytecode = True

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2 as cv

import config as cf

def list_dataset(img_dir_path):
    img_paths = sorted([p for p in img_dir_path.glob("**/*") if p.suffix.lower() in cf.img_ext])

    labels = []
    class_names = [0] * cf.classesSize
    for p in img_paths:
        parent_dir_name = p.parent.name
        cls_id = int(parent_dir_name.split("_")[0]) # フォルダ名を _ で分割した先頭をIDにする
        labels.append(cls_id) # 最初の要素をクラスIDとして使用
        if parent_dir_name not in class_names:
            class_names[cls_id] = parent_dir_name # クラス名のリストを作る

    class_to_idx = {class_names[i]: i for i in range(cf.classesSize)}
    return img_paths, labels, class_to_idx

class ImageFolder_directory(Dataset):
    def __init__(self, paths, labels, indices, transform, class_to_idx):
        self.paths = paths
        self.labels = labels
        self.indices = indices
        self.transform = transform
        self.class_to_idx = class_to_idx

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img_path = self.paths[real_idx]
        img = cv.cvtColor(cv.imread(str(img_path)), cv.COLOR_BGR2RGB)

        img = self.transform(img)
        return img, self.labels[real_idx]

    def __len__(self): # ディレクトリ内の画像ファイルの数
        return len(self.indices)

if __name__ == "__main__":
    import time, pathlib
    f_tm = time.time()

    paths, labels, class_to_idx = list_dataset(pathlib.Path(sys.argv[1]))

    dataset_raw = ImageFolder_directory(paths, labels, list(range(len(paths))), cf.transforms_train, class_to_idx)

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
