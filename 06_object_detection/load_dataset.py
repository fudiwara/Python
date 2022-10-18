import sys, os, random
sys.dont_write_bytecode = True
import pathlib
import numpy as np
from PIL import Image, ImageDraw
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import config as cf

def read_csv_split_key(csv_path, s_rate):
    df = pd.read_csv(csv_path) # ベースのCSVファイル読み込み
    img_filenames = df.image.unique().tolist() # "image"列のリスト作成
    random.shuffle(img_filenames)
    list_A = img_filenames[:int(len(img_filenames) * s_rate)] # とある割合でランダムにリスト作成

    df_a, df_b = df.copy(), df.copy()
    del_a, del_b = [], []
    for i in range(len(df)):
        file_name = df["image"][i]
        if file_name in list_A: del_b.append(i) # 削除用のリスト作成
        else: del_a.append(i)
    df_a.drop(index=df_a.index[del_a], inplace=True) # dropメソッドでまとめて行を削除
    df_b.drop(index=df_b.index[del_b], inplace=True)
    return df, df_a, df_b

class ImageFolderAnnotationRect(Dataset):
    def __init__(self, imgs_dir, annotations_dataframe): # 画像ファイルのパス一覧
        self.imgs_dir = pathlib.Path(imgs_dir)
        self.df = annotations_dataframe
        self.img_filenames = self.df.image.unique().tolist()

    def __len__(self) -> int:
        return len(self.img_filenames)

    def get_image_and_labels_by_idx(self, index):
        image_name = self.img_filenames[index]
        # print(image_name)
        image = Image.open(self.imgs_dir / image_name).convert("RGB")
        pascal_bboxes = self.df[self.df.image == image_name][
            ["xmin", "ymin", "xmax", "ymax"]
        ].values
        # class_labels = self.df[self.df.image == image_name][["class"]].values
        class_labels = np.ones(len(pascal_bboxes))
        return image, pascal_bboxes, class_labels, index, image_name
    
    def save_image(self, index):
        image, bboxes, class_labels, image_id, image_name = self.get_image_and_labels_by_idx(index)
        draw = ImageDraw.Draw(image)
        bs = bboxes.tolist()
        for b in bs: draw.rectangle((b), outline=(255, 0, 0))
        save_filename = f"_s_{image_id:04}.png"
        image.save(save_filename)

def get_train_transforms(target_img_size=cf.imageSize):
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Resize(height=target_img_size, width=target_img_size, p=1),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2(p=1),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )

def get_valid_transforms(target_img_size=cf.imageSize):
    return A.Compose(
        [
            A.Resize(height=target_img_size, width=target_img_size, p=1),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2(p=1),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )

class EfficientDetDataset(Dataset):
    def __init__(self, dataset_adaptor, transforms=get_valid_transforms()):
        self.ds = dataset_adaptor
        self.transforms = transforms

    def __getitem__(self, index):
        (image, pascal_bboxes, class_labels, image_id, image_name) = self.ds.get_image_and_labels_by_idx(index)
        # print(f"id: {image_id} {image_name}")
        # print(pascal_bboxes)
        sample = {
            "image": np.array(image, dtype=np.float32),
            "bboxes": pascal_bboxes,
            "labels": class_labels,
        }

        sample = self.transforms(**sample)
        sample["bboxes"] = np.array(sample["bboxes"])
        image = sample["image"]
        labels = sample["labels"]

        _, new_h, new_w = image.shape
        sample["bboxes"][:, [0, 1, 2, 3]] = sample["bboxes"][
            :, [1, 0, 3, 2]
        ]  # convert to yxyx

        target = {
            "bboxes": torch.as_tensor(sample["bboxes"], dtype=torch.float32),
            "labels": torch.as_tensor(labels),
            "image_id": torch.tensor([image_id]),
            "img_size": (new_h, new_w),
            "img_scale": torch.tensor([1.0]),
        }
        return image, target, image_id

        # y = {"bboxes": torch.as_tensor(sample["bboxes"], dtype=torch.float32), "labels": torch.as_tensor(labels),}
        # return image, y

    def __len__(self):
        return len(self.ds)

def collate_fn_cstm(batch): # 返却値は処理時に整合性が取れるように調整が必要
    images, targets, image_ids = tuple(zip(*batch))
    images = torch.stack(images)
    images = images.float()

    boxes = [target["bboxes"].float() for target in targets]
    labels = [target["labels"].float() for target in targets]
    img_size = torch.tensor([target["img_size"] for target in targets]).float()
    img_scale = torch.tensor([target["img_scale"] for target in targets]).float()

    annotations = {
        "bbox": boxes,
        "cls": labels,
        "img_size": img_size,
        "img_scale": img_scale,
    }
    # return images, annotations, targets, image_ids
    return images, annotations

    # y = {"bbox": boxes, "cls": labels,}
    # return images, y

if __name__ == "__main__":
    # python load_dataset.py /Users/tfuji/work/dataset/carsobj/training_images /Users/tfuji/work/dataset/carsobj/train.csv
    # python load_dataset.py /work/hep/woker_20220822 /work/hep/_train_list_woker_20220822.txt

    import time
    f_tm = time.time()

    train_data_path = sys.argv[1]
    train_annotation_path = sys.argv[2]
    df = pd.read_csv(train_annotation_path)
    train_ds = ImageFolderAnnotationRect(train_data_path, df)
    # for i in range(len(train_ds)): train_ds.save_image(i)

    train_datasets = EfficientDetDataset(train_ds, transforms=get_train_transforms())
    val_datasets = EfficientDetDataset(train_ds)

    train_loader = DataLoader(train_datasets, batch_size=cf.batchSize, shuffle=True, num_workers = os.cpu_count(), pin_memory=True, drop_last=True, collate_fn=collate_fn_cstm,)
    val_loader = DataLoader(val_datasets, batch_size=cf.batchSize, shuffle=True, num_workers = os.cpu_count(), pin_memory=True, drop_last=True, collate_fn=collate_fn_cstm,)
    print(f"train size: {len(train_datasets)}, val size: {len(val_datasets)}")
    print(f"train batch: {len(train_loader)}, val batch: {len(val_loader)}")

    print("train", len(train_loader))
    for n, (imgs, label) in enumerate(train_loader):
    #     # print(labels[n], imgpaths[n])
        print("itr: ", n)
        print(imgs.shape)
        # print(label.shape)
        print(label)
        # if n == 3: break

    print("val", len(val_loader))
    for n, (imgs, label) in enumerate(val_loader):
    #     # print(labels[n], imgpaths[n])
        print("itr: ", n)
        print(imgs.shape)
        # print(label.shape)
        print(label)
        # if n == 3: break

    print()
    print(time.time() - f_tm)