import sys
sys.dont_write_bytecode = True
import pathlib

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import config as cf

class ImageFolder_reg1(Dataset):
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
        reg_val = float(fname_prt[cf.sep_val]) / cf.val_rate
        out_val = torch.tensor(reg_val)

        return img, out_val

    def _get_img_paths(self, img_dir): # 指定ディレクトリ内の画像ファイルパス一覧
        fileList = list(pathlib.Path(img_dir).iterdir())
        i_p = []
        for i in range(len(fileList)):
            if fileList[i].is_file() and (fileList[i].suffix in cf.ext):
                face_atr = fileList[i].stem.split('_')

                if len(face_atr) == cf.sep_num:
                    if face_atr[cf.sep_val].isdecimal():
                        i_p.append(fileList[i])
        
        return i_p

    def __len__(self): # ディレクトリ内の画像ファイルの数
        return len(self.img_paths)


if __name__ == "__main__":
    import time
    f_tm = time.time()

    transform = T.Compose([T.Resize(256), T.ToTensor()])

    dataset_raw = ImageFolder_reg1(sys.argv[1], transform)

    dataloader = DataLoader(dataset_raw, batch_size=3)

    for n, (data, label) in enumerate(dataloader):
        # print(labels[n], imgpaths[n])
        print(n)
        print(data.shape)
        print(label)
        print(label.shape)
        if n == 200: break

    print()
    print(time.time() - f_tm)
