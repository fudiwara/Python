import sys, os, time, pathlib
sys.dont_write_bytecode = True
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

import config as cf
import load_dataset_addInfo as ld

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
model_path = sys.argv[1] # モデルのパス
dataset_path = pathlib.Path(sys.argv[2]) # テスト用の画像が入ったディレクトリのパス
bs = 10

# モデルの定義と読み込みおよび評価用のモードにセットする
model = cf.build_model("eval").to(DEVICE)
model.load_state_dict(torch.load(model_path, map_location = DEVICE, weights_only = False))
model.eval()

# データの読み込み (バッチサイズは適宜変更する)
test_data = ld.ImageFolder_reg2(dataset_path, cf.transforms_eval) # データの読み込み
test_loader = DataLoader(test_data, batch_size = bs, num_workers = os.cpu_count())

label_list_0, pred_list_0, label_list_1, pred_list_1 = [], [], [], []
for i, (data, lbls_0, lbls_1) in enumerate(test_loader):
    data = data.to(DEVICE, non_blocking = True)
    with torch.no_grad(): # 推定のために勾配計算の無効化モードで
        out0, out1 = model(data)
    lbls_0 = lbls_0.detach().cpu().tolist()
    lbls_1 = lbls_1.detach().cpu().tolist()
    pred_0 = out0.detach().cpu().tolist()
    pred_1 = out1.argmax(dim=1).detach().cpu().tolist()

    label_list_0.extend(lbls_0)
    label_list_1.extend(lbls_1)
    pred_list_0.extend(pred_0)
    pred_list_1.extend(pred_1)

    print(f"\r dataset loading: {i + 1} / {len(test_loader)}", end="", flush=True)
print()

val_a_gt_list = np.array(label_list_0) * cf.val_rate_0
val_a_es_list = np.array(pred_list_0) * cf.val_rate_0

val_a_abs_dist = []
f = open(f"_plot{pathlib.Path(model_path).stem}.csv", mode = "w")
for i in range(len(label_list_0)):
    dist_a = val_a_gt_list[i] - val_a_es_list[i]
    val_a_abs_dist.append(abs(dist_a))

    f.write(f"{val_a_gt_list[i]},{val_a_es_list[i]},{dist_a}\n")
f.close()

val_a_abs_dist = np.array(val_a_abs_dist)

print(np.mean(val_a_gt_list), np.var(val_a_gt_list), np.min(val_a_gt_list), np.max(val_a_gt_list))
print(np.mean(val_a_es_list), np.var(val_a_es_list), np.min(val_a_es_list), np.max(val_a_es_list))
print(np.mean(val_a_abs_dist), np.var(val_a_abs_dist), np.min(val_a_abs_dist), np.max(val_a_abs_dist))
cor = np.corrcoef(val_a_gt_list, val_a_es_list)
print(cor[0, 1])

print("---")
print(accuracy_score(label_list_1, pred_list_1)) # 正解率
print(confusion_matrix(label_list_1, pred_list_1)) # 混同行列
print(classification_report(label_list_1, pred_list_1)) # 各種評価指標
