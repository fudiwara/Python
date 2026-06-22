import sys, time, os
sys.dont_write_bytecode = True
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
log_dir = "_log_" + id_str
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

model = cf.GeneratorAE().to(DEVICE)
model = nn.DataParallel(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, betas=(0.9, 0.999), weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cf.epochSize, eta_min=1.0e-6)

# Lab版はまずシンプルにab回帰(L1)
l1_loss = nn.L1Loss()

dataset = ld.load_datasets(dataset_path)
itr_size = max(1, cf.dataset_size // cf.batchSize)
s_tm = time.time()

with open(path_log, mode="w") as f:
    print("loss_ab,lr", file=f)

for i in range(cf.epochSize):
    model.train()
    log_loss = []
    n_tm = time.time()

    for n, (real_ab, imgs_L) in enumerate(dataset):
        real_ab = real_ab.to(DEVICE)  # 2ch
        imgs_L = imgs_L.to(DEVICE)    # 1ch

        pred_ab = model(imgs_L)

        loss = l1_loss(pred_ab, real_ab)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log_loss.append(loss.item())

        print(
            f"\r {i+1:03}/{cf.epochSize:03} [{n+1:04}/{itr_size:04}] "
            f"Lab-L1:{loss.item():.04f}",
            end=""
        )

    scheduler.step()
    current_lr = optimizer.param_groups[0]["lr"]
    m_loss = statistics.mean(log_loss)

    print(
        f"\r {i+1:03}/{cf.epochSize:03} [{n+1:04}/{itr_size:04}] "
        f"Lab-L1:{m_loss:.04f} lr:{current_lr:.7f} {time.time()-n_tm:.01f}s"
    )

    with open(path_log, mode="a") as f:
        print(f"{m_loss},{current_lr}", file=f)

    # 可視化保存: L + pred_ab -> RGB
    with torch.no_grad():
        b = min(imgs_L.shape[0], 16)

        # すべてCPUへ揃える
        L_vis = imgs_L[:b].detach().cpu()                # [-1,1], Bx1xHxW
        ab_vis = pred_ab[:b].detach().clamp(-1, 1).cpu() # [-1,1], Bx2xHxW

        # L: [-1,1] -> [0,255], ab: [-1,1] -> [0,255]
        L_np = (L_vis + 1.0) * 0.5 * 255.0
        ab_np = ab_vis * 127.0 + 128.0
        lab = torch.cat([L_np, ab_np], dim=1)  # CPU tensor, Bx3xHxW

        # 左: 入力L(3ch化) / 右: Lab擬似表示
        left = L_vis.repeat(1, 3, 1, 1)  # CPU
        right = torch.clamp((lab / 255.0) * 2.0 - 1.0, -1, 1)  # CPU
        preview = torch.cat([left, right], dim=0)  # CPU同士なのでOK

        torchvision.utils.save_image(
            preview,
            f"{log_dir}/_e_{i+1:03}.png",
            value_range=(-1.0, 1.0),
            normalize=True
        )

torch.save(model.state_dict(), f"{log_dir}/_ae_lab_{cf.epochSize:03}.pth")
print(f"done {time.time()-s_tm:.01f}s")