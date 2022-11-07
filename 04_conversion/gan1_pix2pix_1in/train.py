import sys, time, os
sys.dont_write_bytecode = True
import numpy as np
import statistics

import torch
from torch import nn
import torchvision

import load_dataset as ld
import config as cf

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
torch.backends.cudnn.benchmark = True
id_str = sys.argv[1]
dataset_path = sys.argv[2]
path_log = "_l_" + id_str + ".csv"
log_dir = "log_" + id_str
if not os.path.exists(log_dir): os.mkdir(log_dir) # モデルの保存用のフォルダ

model_G, model_D = cf.Generator(), cf.Discriminator()
model_G, model_D = nn.DataParallel(model_G), nn.DataParallel(model_D)
model_G, model_D = model_G.to(DEVICE), model_D.to(DEVICE)

params_G = torch.optim.Adam(model_G.parameters(), lr=0.0002, betas=(0.5, 0.999))
params_D = torch.optim.Adam(model_D.parameters(), lr=0.0002, betas=(0.5, 0.999))

# ロスを計算するためのラベル変数 (PatchGAN)
ones = torch.ones(cf.batchSize, 1, 3, 3).to(DEVICE)
zeros = torch.zeros(cf.batchSize, 1, 3, 3).to(DEVICE)

# 損失関数
bce_loss = nn.BCEWithLogitsLoss()
mae_loss = nn.L1Loss()

# 訓練
dataset = ld.load_datasets(dataset_path)
itr_size = cf.dataset_size // cf.batchSize
s_tm = time.time()
log_list = [] # エラー推移の記録用
for i in range(cf.epochSize):
    log_loss_G_sum, log_loss_G_bce, log_loss_G_mae, log_loss_D = [], [], [], []
    n_tm = time.time()
    for n, (real_target, imgs_src) in enumerate(dataset):
        batch_len = len(real_target)
        real_target, imgs_src = real_target.to(DEVICE), imgs_src.to(DEVICE)
        
        fake_target = model_G(imgs_src) # Gの訓練: 偽のターゲットドメイン画像を作成
        fake_target_tensor = fake_target.detach() # 偽画像を一時保存

        # 偽画像を本物と騙せるようにロスを計算
        LAMBD = 100.0 # BCEとMAEの係数
        out = model_D(torch.cat([fake_target, imgs_src], dim=1))
        loss_G_bce = bce_loss(out, ones[:batch_len])
        loss_G_mae = LAMBD * mae_loss(fake_target, real_target)
        loss_G_sum = loss_G_bce + loss_G_mae

        log_loss_G_bce.append(loss_G_bce.item())
        log_loss_G_mae.append(loss_G_mae.item())
        log_loss_G_sum.append(loss_G_sum.item())

        # 微分計算・重み更新
        params_D.zero_grad()
        params_G.zero_grad()
        loss_G_sum.backward()
        params_G.step()

        # Discriminatoの訓練: 本物のターゲットドメイン画像を本物と識別できるようにロスを計算
        real_out = model_D(torch.cat([real_target, imgs_src], dim=1))
        loss_D_real = bce_loss(real_out, ones[:batch_len])

        # 偽の画像の偽と識別できるようにロスを計算
        fake_out = model_D(torch.cat([fake_target_tensor, imgs_src], dim=1))
        loss_D_fake = bce_loss(fake_out, zeros[:batch_len])

        # 実画像と偽画像のロスを合計
        loss_D = loss_D_real + loss_D_fake
        log_loss_D.append(loss_D.item())

        # 微分計算・重み更新
        params_D.zero_grad()
        params_G.zero_grad()
        loss_D.backward()
        params_D.step()

        print("\r {:03} / {:03} [ {:04} / {:04} ] GL bce: {:.04f} l1: {:.04f} DL: {:.04f}".format(i + 1, cf.epochSize, n, itr_size, loss_G_bce.item(), loss_G_mae.item(), loss_D.item()), end = "")
        # if n == 3: break

    gl_mean = statistics.mean(log_loss_G_sum)
    gl_bce = statistics.mean(log_loss_G_bce)
    gl_l1 = statistics.mean(log_loss_G_mae)
    dl = statistics.mean(log_loss_D)
    log_list.append([gl_mean, gl_bce, gl_l1, dl])

    print("\r {:03} / {:03} [ {:04} / {:04} ] GL bce: {:.04f} l1: {:.04f} DL: {:.04f} {:.01f}s".format(i + 1, cf.epochSize, n, itr_size, gl_bce, gl_l1, dl, time.time() - n_tm))

    # Gでの生成画像例とソース画像を連結してから保存
    buf_save_imgs = torch.cat([fake_target_tensor[:min(batch_len, 32)], real_target[:min(batch_len, 32)]], dim=0)
    torchvision.utils.save_image(buf_save_imgs, f"{log_dir}/_e_{i + 1:03}.png", value_range=(-1.0,1.0), normalize=True)

    # モデルの保存
    if 0 < i and (i % 10 == 0 or i == cf.epochSize - 1):
        torch.save(model_G.state_dict(), f"{log_dir}/_gen_{i + 1:03}.pth")
        torch.save(model_D.state_dict(), f"{log_dir}/_dis_{i + 1:03}.pth")
    
np.savetxt(path_log, np.array(log_list), delimiter = ",", fmt="%.5f") # ログの保存
print("done %.0fs" % (time.time() - s_tm))
