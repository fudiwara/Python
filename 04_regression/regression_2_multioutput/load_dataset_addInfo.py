import sys
sys.dont_write_bytecode = True

import torch
from torch.utils.data import Dataset, DataLoader
import cv2 as cv

import config as cf

def list_dataset(img_dir_path):
    fileList = sorted([p for p in img_dir_path.iterdir() if p.suffix.lower() in cf.img_ext])
    img_paths, labels_0, labels_1 = [], [], []
    for i in range(len(fileList)):
        parts_atrs = fileList[i].stem.split('_')

        if len(parts_atrs) == cf.sep_num:
            if parts_atrs[cf.sep_val_0].isdecimal() and parts_atrs[cf.sep_val_1].isdecimal():
                img_paths.append(fileList[i])
                labels_0.append(float(parts_atrs[cf.sep_val_0]) / cf.val_rate_0)
                labels_1.append(int(parts_atrs[cf.sep_val_1])) # 男: 0、 女: 1
    return img_paths, labels_0, labels_1

class ImageFolder_reg2(Dataset):
    def __init__(self, img_paths, labels_0, labels_1, indices, transform):
        self.img_paths = img_paths
        self.labels_0 = labels_0
        self.labels_1 = labels_1
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img_path = self.img_paths[real_idx]
        img = cv.cvtColor(cv.imread(str(img_path)), cv.COLOR_BGR2RGB)
        img = self.transform(img)

        out_val_0 = torch.tensor(self.labels_0[real_idx], dtype = torch.float32) # 年齢
        out_val_1 = torch.tensor(self.labels_1[real_idx], dtype = torch.long) # 識別用のテンソル 男: 0、 女: 1

        return img, out_val_0, out_val_1

    def __len__(self): # ディレクトリ内の画像ファイルの数
        return len(self.img_paths)

if __name__ == "__main__":
    import time, pathlib
    f_tm = time.time()

    img_paths, labels_0, labels_1 = list_dataset(pathlib.Path(sys.argv[1]))
    dataset_raw = ImageFolder_reg2(img_paths, labels_0, labels_1, list(range(len(img_paths))), cf.transforms_train) # データの読み込み

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