import sys, os
sys.dont_write_bytecode = True
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import numpy as np
import config as cf
import load_dataset_addInfo as ld

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
model_path = sys.argv[1] # モデルのパス
dataset_path = sys.argv[2] # テスト用の画像が入ったディレクトリのパス
bs = 10

# モデルの定義と読み込みおよび評価用のモードにセットする
model = cf.build_model().to(DEVICE)
if DEVICE == "cuda": model.load_state_dict(torch.load(model_path))
else: model.load_state_dict(torch.load(model_path, torch.device("cpu")))
model.eval()

# データの読み込み (バッチサイズは適宜変更する)
data_transforms = T.Compose([T.Resize(cf.cellSize), T.CenterCrop(cf.cellSize), T.ToTensor()])
test_data = ld.ImageFolder_reg2(dataset_path, data_transforms) # データの読み込み
test_loader = DataLoader(test_data, batch_size = bs, num_workers = os.cpu_count())

label_list_0, pred_list_0, label_list_1, pred_list_1 = [], [], [], []
for i, (data, lbls) in enumerate(test_loader):
    data = data.to(DEVICE)
    outputs = model(data)
    lbls = lbls.to("cpu").detach().numpy().tolist()
    pred = outputs.to("cpu").detach().numpy().tolist()

    for b in range(len(lbls)):
        lbl_0 = lbls[b][0] * cf.val_rate_0
        lbl_1 = lbls[b][1] * cf.val_rate_1
        pred_val_0 = pred[b][0] * cf.val_rate_0
        pred_val_1 = pred[b][1] * cf.val_rate_1
        
        label_list_0.append(lbl_0)
        label_list_1.append(lbl_1)
        pred_list_0.append(pred_val_0)
        pred_list_1.append(pred_val_1)

    print(f"\r dataset loading: {i + 1} / {len(test_loader)}", end="", flush=True)
print()

val_a_gt_list = np.array(label_list_0)
val_a_es_list = np.array(pred_list_0)
val_b_gt_list = np.array(label_list_1)
val_b_es_list = np.array(pred_list_1)

val_a_abs_dist, val_b_abs_dist = [], []
f = open("_plot_rn.csv", mode = "w")
for i in range(len(label_list_0)):
    dist_a = val_a_gt_list[i] - val_a_es_list[i]
    dist_b = val_b_gt_list[i] - val_b_es_list[i]
    val_a_abs_dist.append(abs(dist_a))
    val_b_abs_dist.append(abs(dist_b))

    f.write(f"{val_a_gt_list[i]},{val_a_es_list[i]},{dist_a},{val_b_gt_list[i]},{val_b_es_list[i]},{dist_b}\n")
f.close()

val_a_abs_dist = np.array(val_a_abs_dist)
val_b_abs_dist = np.array(val_b_abs_dist)

print(np.mean(val_a_gt_list), np.var(val_a_gt_list), np.min(val_a_gt_list), np.max(val_a_gt_list))
print(np.mean(val_a_es_list), np.var(val_a_es_list), np.min(val_a_es_list), np.max(val_a_es_list))
print(np.mean(val_a_abs_dist), np.var(val_a_abs_dist), np.min(val_a_abs_dist), np.max(val_a_abs_dist))

print(np.mean(val_b_gt_list), np.var(val_b_gt_list), np.min(val_b_gt_list), np.max(val_b_gt_list))
print(np.mean(val_b_es_list), np.var(val_b_es_list), np.min(val_b_es_list), np.max(val_b_es_list))
print(np.mean(val_b_abs_dist), np.var(val_b_abs_dist), np.min(val_b_abs_dist), np.max(val_b_abs_dist))

