import sys
sys.dont_write_bytecode = True

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import cv2 as cv

import config as cf

class ImageFolder_reg1(Dataset):
    def __init__(self, img_dir, transform): # 画像ファイルのパス一覧
        self.img_paths = self._get_img_paths(img_dir)
        self.transform = transform

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = cv.cvtColor(cv.imread(str(path)), cv.COLOR_BGR2RGB)
        img = self.transform(img)

        fname_prt = path.stem.split("_")
        # print(fname_prt)
        reg_val = float(fname_prt[cf.sep_val]) / cf.val_rate
        out_val = torch.tensor([reg_val]) # 学習時の処理における次元数を考えて配列の中の数値にしておく

        return img, out_val

    def _get_img_paths(self, img_dir): # 指定ディレクトリ内の画像ファイルパス一覧
        fileList = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in cf.ext])
        i_p = []
        for i in range(len(fileList)):
            face_atrs = fileList[i].stem.split('_')

            if len(face_atrs) == cf.sep_num:
                if face_atrs[cf.sep_val].isdecimal():
                    i_p.append(fileList[i])
        return i_p

    def __len__(self): # ディレクトリ内の画像ファイルの数
        return len(self.img_paths)

if __name__ == "__main__":
    import time, pathlib
    f_tm = time.time()

    dataset_raw = ImageFolder_reg1(pathlib.Path(sys.argv[1]), cf.transforms_train) # データの読み込み

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
