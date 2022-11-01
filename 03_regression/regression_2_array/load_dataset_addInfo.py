import sys
sys.dont_write_bytecode = True
import pathlib

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import config as cf

def genGridMask(src_pil):
    if 0.8 < np.random.uniform(): return src_pil # この割合で動作する
    p_mask = 0.8 # グリッド内でマスクされない割合
    r_rate = 0.6

    src = np.array(src_pil, dtype=np.uint8)
    if src.shape[0] < src.shape[1] : side = src.shape[0]
    else: side = src.shape[1]

    d = np.random.randint(side // 7, side)
    r = int(r_rate * d)
    start_rx, start_ry = np.random.randint(0, r), np.random.randint(0, r)

    mask = np.ones((src.shape[0]+d, src.shape[1]+d, 3), dtype=np.uint8)
    for i in range(start_ry, src.shape[0]+d, d):
        for j in range(start_rx, src.shape[1]+d, d):
            if p_mask < np.random.uniform():
                mask[i: i+(d-r), j: j+(d-r)] = 0

    mask = mask[:src.shape[0], :src.shape[1]]
    dst = src * mask

    dst_pil = Image.fromarray(dst) # ここまでnp形式だったのでPILに戻す
    return dst_pil

class ImageFolder_reg2(Dataset):
    def __init__(self, img_dir, transform = None): # 画像ファイルのパス一覧
        self.img_paths = self._get_img_paths(img_dir)
        self.transform = transform

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = Image.open(path).convert("RGB") # 画像読み込み
        img = genGridMask(img)
        img = self.transform(img)

        fname_prt = path.stem.split("_")
        reg_val_0 = float(fname_prt[cf.sep_val_0]) / cf.val_rate_0
        reg_val_1 = float(fname_prt[cf.sep_val_1]) / cf.val_rate_1
        out_val = np.array([reg_val_0, reg_val_1]) # それぞれの真値をまとめて配列にする

        return img, torch.tensor(out_val.astype(np.float32))

    def _get_img_paths(self, img_dir): # 指定ディレクトリ内の画像ファイルパス一覧
        fileList = list(pathlib.Path(img_dir).iterdir())
        # fileList.sort() 連番に処理したい場合はいれておく
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
