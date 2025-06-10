import sys, random
sys.dont_write_bytecode = True
import json
import pathlib

import numpy as np
from PIL import Image

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO

import config as cf
import pyt_det.transforms as T

def collate_fn(batch):
    return tuple(zip(*batch))

def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor()) # PIL imageをPyTorch Tensorに変換
    transforms.append(T.ConvertImageDtype(torch.float))
    if train: transforms.append(T.RandomHorizontalFlip(0.5)) # 訓練中はランダムで水平に反転
    return T.Compose(transforms)

class loadImagesCocoJson(Dataset):
    def __init__(self, img_dir_path, annot_json_path, transforms): # 画像とアノテーションjsonへのパス
        coco = COCO(annot_json_path)
        img_idxs, img_paths = [], []
        with open(annot_json_path) as f:
            for img_data in json.load(f)["images"]:
                id = img_data["id"]
                if 0 < len(coco.getAnnIds(id)):
                    img_idxs += [img_data["id"]]
                    img_paths.append(pathlib.Path(img_dir_path) / img_data["file_name"])
        self.img_idxs = img_idxs
        self.img_paths = img_paths
        self.coco = coco
        self.transforms = transforms

    def __len__(self):
        return len(self.img_idxs)

    def __getitem__(self, idx):
        # print(img_path, mask_path)
        img = Image.open(self.img_paths[idx]).convert("RGB") # 画像はそのままひらく
        img_idx = self.img_idxs[idx]
        anns = self.coco.loadAnns(self.coco.getAnnIds(img_idx)) # 同じIDのアノテーションデータ

        boxes = np.array([x["bbox"] for x in anns])
        labels = [x["category_id"] for x in anns]
        areas = [box[2] * box[3] for box in boxes]
        iscrowds = [x["iscrowd"] for x in anns]
        masks = [self.coco.annToMask(x) for x in anns]

        targets = {}
        boxes = torch.as_tensor(boxes, dtype = torch.float32)
        targets["boxes"] = torchvision.ops.box_convert(boxes,"xywh","xyxy")
        targets["labels"] = torch.as_tensor(labels, dtype = torch.int64)
        targets["masks"] = torch.as_tensor(np.array(masks), dtype = torch.uint8)
        targets["image_id"] = torch.tensor([img_idx])
        targets["area"] = torch.tensor(areas)
        targets["iscrowd"] = torch.as_tensor(iscrowds, dtype = torch.int64)

        img, targets = self.transforms(img, targets)

        return img, targets

if __name__ == "__main__":
    import cv2

    img_dir_path = pathlib.Path(sys.argv[1])
    annot_json_path = sys.argv[2]
    img_dir = pathlib.Path("_test")

    train_dataset = loadImagesCocoJson(img_dir_path, annot_json_path, get_transform(train=True))

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)

    if(not img_dir.exists()): img_dir.mkdir() # 出力用のディレクトリ生成

    print("don pre-pro")
    for n, (imgs, lbls) in enumerate(train_loader):
        for j in range(len(lbls)):
            img_id = lbls[j]["image_id"].item()
            # print(img_id)
            
            tmp = imgs[j]
            tmp = tmp.permute(1, 2, 0) # 画像出力用に次元の入れ替え
            tmp = torch.clamp(tmp, min=0, max=1)
            tmp = tmp.to("cpu").detach().numpy() # np配列に変換
            img_tmp = (tmp * 255).astype(np.uint8) # 0-1の範囲なので255倍して画像用データへ
            img_src = cv2.cvtColor(img_tmp, cv2.COLOR_RGB2BGR)

            _, iw, ih = imgs[j].shape
            masks = lbls[j]["masks"]
            cat_id = lbls[j]["labels"]
            bboxs = lbls[j]["boxes"]
            for i in range(len(masks)):
                mw, mh = masks[i].shape
                mask = masks[i]
                cid = cat_id[i] % len(cf.box_col)
                rm = mask * cf.box_col[cid][0]
                gm = mask * cf.box_col[cid][1]
                bm = mask * cf.box_col[cid][2]

                b = bboxs[i]
                p0, p1 = (int(b[0]), int(b[1])), (int(b[2]), int(b[3]))
                col_m = cv2.merge([bm.to("cpu").detach().numpy(), gm.to("cpu").detach().numpy(), rm.to("cpu").detach().numpy()])
                img_src = cv2.addWeighted(img_src, 1, col_m, 1, 0)
                cv2.rectangle(img_src, p0, p1, cf.box_col[cid], thickness = 2)

                # print(mw, mh)
                # if iw != mw or ih != mh:
                #     print(n, lbls[0]["image_id"])

            output_filepath = img_dir / f"{img_id:07}.jpg"
            cv2.imwrite(str(output_filepath), img_src)

        # if n == 10: break

    """
    for n, (imgs, lbls) in enumerate(val_loader):
    #     # print(labels[n], imgpaths[n])
        print("itr: ", n)
        print(imgs)
        # print(label.shape)
        print(lbls)
        # if n == 0: break
    """
