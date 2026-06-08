import sys
sys.dont_write_bytecode = True

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import cv2 as cv
import numpy as np

import config as cf

class ImageFolder_reg2(Dataset):
    def __init__(self, img_dir, transform = None): # 画像ファイルのパス一覧
        self.img_paths = self._get_img_paths(img_dir)
        self.transform = transform

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = cv.cvtColor(cv.imread(str(path)), cv.COLOR_BGR2RGB)
        img = self.transform(img)

        # print(str(path))
        fname_prt = path.stem.split("_")
        reg_val_0 = float(fname_prt[cf.sep_val_0]) / cf.val_rate_0
        out_val_0 = torch.tensor(reg_val_0, dtype=torch.float32) # 学習時の処理における次元数を考えて配列の中の数値にしておく

        reg_val_1 = int(fname_prt[cf.sep_val_1]) # UTKFaceだと 男: 0、 女: 1
        out_val_1 = torch.tensor(reg_val_1, dtype = torch.long) # 識別用のテンソル

        return img, out_val_0, out_val_1

    def _get_img_paths(self, img_dir): # 指定ディレクトリ内の画像ファイルパス一覧
        fileList = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in cf.ext])
        i_p = []
        for i in range(len(fileList)):
            face_atr = fileList[i].stem.split('_')

            if len(face_atr) == cf.sep_num:
                if face_atr[cf.sep_val_0].isdecimal() and face_atr[cf.sep_val_1].isdecimal():
                    i_p.append(fileList[i])
        return i_p

    def __len__(self): # ディレクトリ内の画像ファイルの数
        return len(self.img_paths)

if __name__ == "__main__":
    import time, pathlib
    f_tm = time.time()

    dataset_raw = ImageFolder_reg2(pathlib.Path(sys.argv[1]), cf.transforms_train) # データの読み込み

    dataloader = DataLoader(dataset_raw, batch_size = 3)

    for n, (data, lbl_0, lbl_1) in enumerate(dataloader):
        # print(labels[n], imgpaths[n])
        print(n)
        print(data.shape)
        print(lbl_0, lbl_1)
        print(lbl_0.shape, lbl_1.shape)
        if n == 200: break

    print()
    print(time.time() - f_tm)