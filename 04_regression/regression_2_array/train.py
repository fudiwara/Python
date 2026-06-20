import sys, time, os, pathlib
sys.dont_write_bytecode = True
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

import config as cf
import load_dataset_addInfo as ld

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
id_str = sys.argv[1]
dataset_path = pathlib.Path(sys.argv[2])
path_log = "_l_" + id_str + ".csv"
log_dir = pathlib.Path("_log_" + id_str)
log_dir.mkdir(exist_ok=True) # モデルの保存用のフォルダ
disp_score_t = ""

paths, labels_0, labels_1 = ld.list_dataset(dataset_path) # データの読み込み

# 学習・検証データを分割
indices = torch.randperm(len(paths)).tolist() # インデックスをランダムシャッフルしたリスト
train_data_size = int(cf.splitRateTrain * len(paths)) # 学習サンプルのサイズ
train_idx, val_idx = indices[:train_data_size], indices[train_data_size:] # 各インデックス
print(len(paths), train_data_size, len(val_idx))

train_loader = DataLoader(ld.ImageFolder_reg2(paths, labels_0, labels_1, train_idx, cf.transforms_train), batch_size = cf.batchSize, num_workers = os.cpu_count(), pin_memory=True, drop_last=True, shuffle = True)
val_loader = DataLoader(ld.ImageFolder_reg2(paths, labels_0, labels_1, val_idx, cf.transforms_train), batch_size = cf.batchSize, num_workers = os.cpu_count(), pin_memory=True, drop_last=True)

# モデル、損失関数、最適化関数、収束率の定義
model = cf.build_model("train").to(DEVICE)
criterion = torch.nn.MSELoss()
calc_acc = torch.nn.L1Loss()
optimizer = torch.optim.AdamW(model.parameters(), lr = 0.0001)

def Train_Eval(model, criterion, optimizer, data_loader, device, epoch, max_epoch, is_val = False):
    total_loss = 0.0
    total_acc = 0.0
    y_true_all, y_pred_all = [], []
    global disp_score_t
    model.eval() if is_val else model.train()
    for n, (data, lbls) in enumerate(data_loader): # バッチ毎にデータ読み込み
        if is_val == False: optimizer.zero_grad()
        data = data.to(device)
        lbls = lbls.to(device)

        if is_val:
            with torch.no_grad():
                outputs = model(data)
        else:
            outputs = model(data)

        loss = criterion(outputs, lbls)
        total_loss += loss.item()
        acc = calc_acc(outputs, lbls)
        total_acc += acc.item()
        
        y_true_all.append(lbls.detach().cpu().view(-1).numpy())
        y_pred_all.append(outputs.detach().cpu().view(-1).numpy())
        
        if is_val == False:
            loss.backward()
            optimizer.step()
        
        if is_val == False:
            disp_score_t = f"{epoch + 1:03} / {max_epoch:03} [ {n + 1:04} / {len(data_loader):04} ] l: {total_loss/(n+1):.05f} a_e: {total_acc/(n+1):.06f}"
            print(f"\r {disp_score_t}", end = "")
        else: 
            print(f"\r {disp_score_t} l: {total_loss/(n+1):.05f} a_e: {total_acc/(n+1):.06f} ", end = "")

        # if n == 1: break
    # if is_val == False: scheduler.step()

    # 学習に使った画像の一部を保存
    torchvision.utils.save_image(data[:min(cf.batchSize, 16)], log_dir / f"_i_{id_str}_{epoch + 1:03}.png", value_range=(-1.0,1.0), normalize=True)

    avg_loss = total_loss / (n + 1)
    avg_acc = total_acc / (n + 1)

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    mae, rmse, r2, corr = cf.calc_reg_metrics(y_true_all, y_pred_all, cf.val_rate_0) # 評価値計算

    return avg_loss, avg_acc, mae, rmse, r2, corr # rmse: 外れ値悪化確認用、r2: 分散量の確認、 corr: 相関係数の確認

best_loss = None
with open(path_log, mode = "w") as f: print("train_loss,val_loss,train_acc,val_acc,train_mae,train_rmse,train_r2,train_corr,val_mae,val_rmse,val_r2,val_corr", file = f)
s_tm = time.time()

for epoch in range(cf.epochSize):
    n_tm = time.time()
    train_loss, train_acc, train_mae, train_rmse, train_r2, train_corr = Train_Eval(model, criterion, optimizer, train_loader, DEVICE, epoch, cf.epochSize)
    val_loss, val_acc, val_mae, val_rmse, val_r2, val_corr = Train_Eval(model, criterion, optimizer, val_loader, DEVICE, epoch, cf.epochSize, is_val=True)
    print(f" {time.time() - n_tm:.0f}s")

    if best_loss is None or val_loss < best_loss: # lossを更新したときのみ保存
        best_loss = val_loss
        torch.save(model.state_dict(), log_dir / f"best.pth") # モデルの保存

    # 学習の状況をCSVに保存
    with open(path_log, mode = "a") as f: print(f"{train_loss},{val_loss},{train_acc},{val_acc},{train_mae},{train_rmse},{train_r2},{train_corr},{val_mae},{val_rmse},{val_r2},{val_corr}", file = f)

torch.save(model.state_dict(), log_dir / f"_m_{id_str}_{cf.epochSize:03}.pth") # モデルの保存
print(f"done {time.time() - s_tm:.0f}s")
