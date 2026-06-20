import sys
sys.dont_write_bytecode = True

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import cv2 as cv

import config as cf

def list_dataset(img_dir_path):
    fileList = sorted([p for p in img_dir_path.iterdir() if p.suffix.lower() in cf.img_ext])
    img_paths, labels = [], []
    for i in range(len(fileList)):
        parts_atrs = fileList[i].stem.split('_')

        if len(parts_atrs) == cf.sep_num:
            if parts_atrs[cf.sep_val].isdecimal():
                img_paths.append(fileList[i])
                labels.append(float(parts_atrs[cf.sep_val]) / cf.val_rate)
    return img_paths, labels

class ImageFolder_reg1(Dataset):
    def __init__(self, img_paths, labels, indices, transform):
        self.img_paths = img_paths
        self.labels = labels
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img_path = self.img_paths[real_idx]
        img = cv.cvtColor(cv.imread(str(img_path)), cv.COLOR_BGR2RGB)
        img = self.transform(img)

        out_val = torch.tensor([self.labels[real_idx]]) # 学習時の処理における次元数を考えて配列の中の数値にしておく

        return img, out_val

    def __len__(self): # ディレクトリ内の画像ファイルの数
        return len(self.indices)

if __name__ == "__main__":
    import time, pathlib
    f_tm = time.time()

    paths, labels = list_dataset(pathlib.Path(sys.argv[1]))
    dataset_raw = ImageFolder_reg1(paths, labels, list(range(len(paths))), cf.transforms_train) # データの読み込み

    dataloader = DataLoader(dataset_raw, batch_size = 3)

    for n, (data, label) in enumerate(dataloader):
        # print(labels[n], imgpaths[n])
        print(n)
        print(data.shape)
        print(label)
        print(label.shape)
        if n == 200: break

    print()
    print(time.time() - f_tm)
