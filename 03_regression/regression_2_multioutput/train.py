import sys, time, os
sys.dont_write_bytecode = True
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import config as cf
import load_dataset_addInfo as ld

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
id_str = sys.argv[1]
dataset_path = sys.argv[2]
path_log = "_l_" + id_str + ".csv"
log_dir = "_log_" + id_str
if not os.path.exists(log_dir): os.mkdir(log_dir) # モデルの保存用のフォルダ
disp_score_t = ""

datasets_raw = ld.ImageFolder_reg2(dataset_path, cf.data_transforms) # データの読み込み

# 学習・検証データを分割
train_data_size = int(cf.splitRateTrain * len(datasets_raw))
val_data_size = len(datasets_raw) - train_data_size
train_dataset, val_dataset = torch.utils.data.random_split(datasets_raw, [train_data_size, val_data_size])
print(len(datasets_raw), train_data_size, val_data_size)

train_loader = DataLoader(train_dataset, batch_size = cf.batchSize, num_workers = os.cpu_count(), pin_memory=True, drop_last=True, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = cf.batchSize, num_workers = os.cpu_count(), pin_memory=True, drop_last=True)

# モデル、損失関数、最適化関数、収束率の定義
model = cf.build_model("train").to(DEVICE)
criterion = nn.MSELoss()
calc_acc = nn.L1Loss()
# optimizer = optim.Adam(model.parameters())
optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)
rate_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

def Train_Eval(model,criterion,optimizer,scheduler,data_loader,device,epoch,max_epoch,is_val = False):
    total_loss = 0.0
    total_acc_0 = 0.0
    total_acc_1 = 0.0
    counter = 0
    global disp_score_t
    model.eval() if is_val else model.train()
    for n, (data, lbl_0, lbl_1) in enumerate(data_loader): # バッチ毎にデータ読み込み
        counter += data.shape[0]
        if is_val == False: optimizer.zero_grad()
        data = data.to(device)
        lbl_0 = lbl_0.to(device)
        lbl_1 = lbl_1.to(device)

        if is_val:
            with torch.no_grad():
                out0, out1 = model(data)
        else:
            out0, out1 = model(data)

        loss0 = criterion(out0, lbl_0)
        loss1 = criterion(out1, lbl_1)
        loss = loss0 + loss1
        total_loss += loss0.item() + loss1.item()
        acc0 = calc_acc(out0, lbl_0)
        acc1 = calc_acc(out1, lbl_1)
        total_acc_0 += acc0.item()
        total_acc_1 += acc1.item()
        
        if is_val == False:
            loss.backward()
            optimizer.step()
        
        if is_val == False:
            disp_score_t = "{:03} / {:03} [ {:04} / {:04} ] l: {:.05f} a_e0: {:.04f} a_e1: {:.04f}".format(epoch+1, max_epoch, n + 1, len(data_loader), total_loss/(n+1), cf.val_rate_0 * total_acc_0/counter, cf.val_rate_1 * total_acc_1/counter)
            print("\r {}".format(disp_score_t), end = "")
        else: 
            print("\r {} l: {:.05f} a_e0: {:.04f} a_e1: {:.04f} ".format(disp_score_t, total_loss/(n+1), cf.val_rate_0 * total_acc_0/counter, cf.val_rate_1 * total_acc_1/counter), end = "")

        # if n == 1: break
    if is_val == False: scheduler.step()

    # 学習に使った画像の一部を保存
    torchvision.utils.save_image(data[:min(cf.batchSize, 16)], f"{log_dir}/_i_{id_str}_{epoch + 1:03}.png", value_range=(-1.0,1.0), normalize=True)

    return total_loss/(n+1), (total_acc_0 + total_acc_1)/counter

best_loss = None
with open(path_log, mode = "w") as f: print("train_loss,val_loss,train_acc,val_acc", file = f)
s_tm = time.time()

for epoch in range(cf.epochSize):
    n_tm = time.time()
    train_loss, train_acc = Train_Eval(model,criterion,optimizer,rate_scheduler,train_loader,DEVICE,epoch,cf.epochSize) 
    val_loss, val_acc = Train_Eval(model,criterion,optimizer,rate_scheduler,val_loader,DEVICE,epoch,cf.epochSize,is_val=True)
    print(" %.0fs" % (time.time() - n_tm))

    # if best_loss is None or val_loss < best_loss: # lossを更新したときのみ保存
    #     best_loss = val_loss
    # 毎エポックモデルの保存する場合 (とりあえずコメントアウト)
    # torch.save(model.state_dict(), f"{log_dir}/_m_{id_str}_{epoch + 1:03}.pth")

    # 学習の状況をCSVに保存
    with open(path_log, mode = "a") as f: print(f"{train_loss},{val_loss},{train_acc},{val_acc}", file = f)

torch.save(model.state_dict(), f"{log_dir}/_m_{id_str}_{cf.epochSize:03}.pth") # モデルの保存
print(f"done {time.time() - s_tm:.0f}s")
