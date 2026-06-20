import sys, time, os, pathlib
sys.dont_write_bytecode = True

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import config as cf
import load_dataset_dirs as ld

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
id_str = sys.argv[1] # 識別用のID
dataset_path = pathlib.Path(sys.argv[2]) # フォルダのあるパス
path_log = "_l_" + id_str + ".csv"
log_dir = pathlib.Path("_log_" + id_str)
log_dir.mkdir(exist_ok = True) # モデルの保存用のフォルダ
disp_score_t = ""

paths, labels, class_to_idx = ld.list_dataset(dataset_path)

# 学習・検証データを分割
indices = torch.randperm(len(paths)).tolist() # インデックスをランダムシャッフルしたリスト
train_data_size = int(cf.splitRateTrain * len(paths)) # 学習サンプルのサイズ
train_idx, val_idx = indices[:train_data_size], indices[train_data_size:] # 各インデックス

train_dataset = ld.ImageFolder_directory(paths, labels, train_idx, cf.transforms_train, class_to_idx)
val_dataset = ld.ImageFolder_directory(paths, labels, val_idx, cf.transforms_eval, class_to_idx)

print(train_dataset.class_to_idx)
print(len(paths), train_data_size, len(val_idx))

train_loader = DataLoader(train_dataset, batch_size = cf.batchSize, num_workers = os.cpu_count(), shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = cf.batchSize, num_workers = os.cpu_count())

# モデル、損失関数、最適化関数、収束率の定義
model = cf.build_model("train").to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr = 0.0001, weight_decay = 1e-8)

def Train_Eval(model, criterion, optimizer, data_loader, device, epoch, max_epoch, is_val = False):
    total_loss = 0.0
    total_acc = 0.0
    counter = 0
    global disp_score_t
    model.eval() if is_val else model.train()
    for n, (data, label) in enumerate(data_loader): # バッチ毎にデータ読み込み
        counter += data.shape[0]
        if is_val == False:
            optimizer.zero_grad()
        data = data.to(device)
        label = label.to(device)
        if is_val:
            with torch.no_grad():
                output = model(data)
        else:
            output = model(data)
        
        loss = criterion(output, label)
        total_loss += loss.item()
        total_acc += cf.calc_acc(output, label).item()

        if is_val == False:
            loss.backward()
            optimizer.step()
        
        # lossおよび精度を表示・1エポックが終わる度にかかった時間も表示する
        if is_val == False:
            disp_score_t = f"{epoch + 1:03} / {max_epoch:03} [ {n + 1:04} / {len(data_loader):04} ] l: {total_loss / (n + 1):.05f} a: {total_acc / counter:.03f}"
            print(f"\r {disp_score_t}", end = "")
        else: 
            print(f"\r {disp_score_t} vl: {total_loss / (n + 1):.05f} va: {total_acc / counter:.03f}", end = "")

    # if is_val == False: rate_scheduler.step() # 学習率の変更

    # 学習に使った画像の一部を保存
    torchvision.utils.save_image(data[:min(cf.batchSize, 16)], log_dir / f"_i_{id_str}_{epoch + 1:03}.png", value_range=(-1.0,1.0), normalize=True)

    return total_loss / (n + 1), total_acc / counter

best_loss = None
with open(path_log, mode = "w") as f: print("train_loss,val_loss,train_acc,val_acc", file = f)
s_tm = time.time()

for epoch in range(cf.epochSize):
    n_tm = time.time()
    train_loss, train_acc = Train_Eval(model, criterion, optimizer, train_loader, DEVICE, epoch, cf.epochSize) 
    val_loss, val_acc = Train_Eval(model, criterion, optimizer, val_loader, DEVICE, epoch, cf.epochSize, is_val = True)
    print(f" {time.time() - n_tm:.0f}s")

    if best_loss is None or val_loss < best_loss: # lossを更新したときのみ保存
        best_loss = val_loss
        torch.save(model.state_dict(), log_dir / "best.pth") # モデルの保存

    # 学習の状況をCSVに保存
    with open(path_log, mode = "a") as f: print(f"{train_loss},{val_loss},{train_acc},{val_acc}", file = f)

torch.save(model.state_dict(), log_dir / f"_m_{id_str}_{cf.epochSize:03}.pth") # モデルの保存
print(f"done {time.time() - s_tm:.0f}s")
