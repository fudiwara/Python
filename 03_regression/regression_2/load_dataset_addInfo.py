import sys
sys.dont_write_bytecode = True
import pathlib

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import config as cf

class ImageFolder_reg2(Dataset):
    def __init__(self, img_dir, transform = None): # 画像ファイルのパス一覧
        self.img_paths = self._get_img_paths(img_dir)
        self.transform = transform
        self.ftf = T.Compose([T.ToTensor()])

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = Image.open(path).convert("RGB") # 画像読み込み
        img = self.transform(img)

        # print(str(path))
        fname_prt = path.stem.split("_")
        # print(fname_prt)
        # age_val = self.ftf(int(fname_prt[cf.sep_val]))
        # print(int(fname_prt[cf.sep_val]))
        reg_val_0 = float(fname_prt[cf.sep_val_0]) / cf.val_rate_0
        out_val_0 = torch.tensor(reg_val_0)

        reg_val_1 = float(fname_prt[cf.sep_val_1]) / cf.val_rate_1
        out_val_1 = torch.tensor(reg_val_1)

        return img, out_val_0, out_val_1

    def _get_img_paths(self, img_dir): # 指定ディレクトリ内の画像ファイルパス一覧
        # img_dir = pathlib.Path(img_dir)
        # img_paths = [p for p in img_dir.iterdir() if p.suffix in ImageFolder_reg1.IMG_EXT]

        fileList = list(pathlib.Path(img_dir).iterdir())
        # fileList.sort()
        i_p = []
        for i in range(len(fileList)):
            if fileList[i].is_file() and (fileList[i].suffix in cf.ext):
                face_atr = fileList[i].stem.split('_')

                if len(face_atr) == cf.sep_num:
                    if face_atr[cf.sep_val_0].isdecimal() and face_atr[cf.sep_val_1].isdecimal():
                        i_p.append(fileList[i])
        
        return i_p

    def __len__(self): # ディレクトリ内の画像ファイルの数
        return len(self.img_paths)


if __name__ == "__main__":
    import time
    f_tm = time.time()

    transform = T.Compose([T.Resize(256), T.ToTensor()])

    dataset_raw = ImageFolder_reg2(sys.argv[1], transform)

    dataloader = DataLoader(dataset_raw, batch_size=4)

    for n, (data, lbl_0, lbl_1) in enumerate(dataloader):
        # print(labels[n], imgpaths[n])
        print(n)
        print(data.shape)
        print(lbl_0)
        print(lbl_0.shape)
        print(lbl_1)
        print(lbl_1.shape)
        if n == 2: break

    print()
    print(time.time() - f_tm)
