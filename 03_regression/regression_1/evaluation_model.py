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

# モデルの定義と読み込みおよび評価用のモードにセットする
model = cf.build_model("eval").to(DEVICE)
if DEVICE == "cuda": model.load_state_dict(torch.load(model_path, weights_only = False))
else: model.load_state_dict(torch.load(model_path, torch.device("cpu")))
model.eval()

# データの読み込み (バッチサイズは適宜変更する)
data_transforms = T.Compose([T.Resize(cf.cellSize), T.CenterCrop(cf.cellSize), T.ToTensor()])
test_data = ld.ImageFolder_reg1(dataset_path, data_transforms) # データの読み込み
test_loader = DataLoader(test_data, batch_size = 20, num_workers = os.cpu_count())

label_list, pred_list = [], []
for i, (data, label) in enumerate(test_loader):
    data = data.to(DEVICE)
    label = label.numpy().tolist()
    with torch.no_grad(): # 推定のために勾配計算の無効化モードで
        outputs = model(data)
    
    out_view = outputs.view(-1, outputs.shape[0])
    out_view = out_view.squeeze()
    pred = out_view.to("cpu").detach().numpy().tolist()
    label_list += label
    pred_list += pred

    print(f"\r dataset loading: {i + 1} / {len(test_loader)}", end="", flush=True)
print()

val_gt_list = np.array(label_list) * cf.val_rate
val_gt_list = val_gt_list.reshape(-1)
val_es_list = np.array(pred_list) * cf.val_rate

val_abs_dist = []
f = open("_plot_rn.csv", mode = "w")
for i in range(len(label_list)):
    dist_age = val_gt_list[i] - val_es_list[i]
    val_abs_dist.append(abs(dist_age))

    f.write(f"{val_gt_list[i]},{val_es_list[i]}\n")
f.close()

val_abs_dist = np.array(val_abs_dist)

print(np.mean(val_gt_list), np.var(val_gt_list), np.min(val_gt_list), np.max(val_gt_list))
print(np.mean(val_es_list), np.var(val_es_list), np.min(val_es_list), np.max(val_es_list))
print(np.mean(val_abs_dist), np.var(val_abs_dist), np.min(val_abs_dist), np.max(val_abs_dist))

cor = np.corrcoef(val_gt_list, val_es_list)
print(cor[0, 1])