import sys, os, pathlib
sys.dont_write_bytecode = True
import torch
from torch.utils.data import DataLoader
import numpy as np

import config as cf
import load_dataset_addInfo as ld

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
model_path = sys.argv[1] # モデルのパス
dataset_path = pathlib.Path(sys.argv[2]) # テスト用の画像が入ったディレクトリのパス

# モデルの定義と読み込みおよび評価用のモードにセットする
model = cf.build_model("eval").to(DEVICE)
model.load_state_dict(torch.load(model_path, map_location = DEVICE, weights_only = False))
model.eval()

# データの読み込み (バッチサイズは適宜変更する)
paths, labels = ld.list_dataset(dataset_path)
test_data = ld.ImageFolder_reg1(paths, labels, list(range(len(paths))), cf.transforms_eval)
test_loader = DataLoader(test_data, batch_size = 16, num_workers = os.cpu_count())

label_list, pred_list = [], []
for i, (data, label) in enumerate(test_loader):
    data = data.to(DEVICE)
    label = label.detach().cpu().tolist()
    with torch.no_grad(): # 推定のために勾配計算の無効化モードで
        outputs = model(data)
    
    pred = outputs.detach().cpu().view(-1).tolist()
    gt = label.detach().cpu().view(-1).tolist()

    label_list += gt
    pred_list += pred

    print(f"\r dataset loading: {i + 1} / {len(test_loader)}", end="", flush=True)
print()

y_true = np.array(label_list).reshape(-1)
y_pred = np.array(pred_list).reshape(-1)

val_gt_list = y_true * cf.val_rate
val_es_list = y_pred * cf.val_rate
dist_list = val_gt_list - val_es_list
val_abs_dist = np.abs(dist_list)

mae, rmse, r2, corr = cf.calc_reg_metrics(y_true, y_pred)

f = open(f"_plot{pathlib.Path(model_path).stem}.csv", mode = "w")
for i in range(len(label_list)):
    dist_age = val_gt_list[i] - val_es_list[i]
    val_abs_dist.append(abs(dist_age))

    f.write(f"{val_gt_list[i]},{val_es_list[i]},{dist_age}\n")
f.close()

val_abs_dist = np.array(val_abs_dist)

print(np.mean(val_gt_list), np.var(val_gt_list), np.min(val_gt_list), np.max(val_gt_list))
print(np.mean(val_es_list), np.var(val_es_list), np.min(val_es_list), np.max(val_es_list))
print(np.mean(val_abs_dist), np.var(val_abs_dist), np.min(val_abs_dist), np.max(val_abs_dist))

cor = np.corrcoef(val_gt_list, val_es_list)
print(cor[0, 1])
print(corr)