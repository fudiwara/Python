import sys, time, os
sys.dont_write_bytecode = True
import statistics

import torch
from torch import nn
import torchvision
import cv2 as cv
import numpy as np

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

# Lab版: Generatorは1ch(L) -> 2ch(ab)
model = cf.GeneratorAE().to(DEVICE)
model = nn.DataParallel(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, betas=(0.9, 0.999), weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=cf.epochSize,
    eta_min=1.0e-6
)

# Lab版はまずL1のみ
l1_loss = nn.L1Loss()

dataset = ld.load_datasets(dataset_path)  # (real_ab, imgs_L)
itr_size = max(1, cf.dataset_size // cf.batchSize)
s_tm = time.time()

with open(path_log, mode="w") as f:
    print("loss_ab,lr,best_loss", file=f)

best_loss = float("inf")
best_epoch = -1
best_path = f"{log_dir}/_ae_best.pth"

for i in range(cf.epochSize):
    model.train()
    log_loss = []
    n_tm = time.time()

    for n, (real_ab, imgs_L) in enumerate(dataset):
        real_ab = real_ab.to(DEVICE)  # [B,2,H,W]
        imgs_L  = imgs_L.to(DEVICE)   # [B,1,H,W]

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

    updated_best = ""
    if m_loss < best_loss:
        best_loss = m_loss
        best_epoch = i + 1
        torch.save(model.state_dict(), best_path)
        updated_best = " *best"

    print(
        f"\r {i+1:03}/{cf.epochSize:03} [{n+1:04}/{itr_size:04}] "
        f"Lab-L1:{m_loss:.04f} best:{best_loss:.04f}(e{best_epoch:03}) "
        f"lr:{current_lr:.7f} {time.time()-n_tm:.01f}s{updated_best}"
    )

    with open(path_log, mode="a") as f:
        print(f"{m_loss},{current_lr},{best_loss}", file=f)

    # プレビュー保存（L + pred_ab -> RGB）
    with torch.no_grad():
        b = min(imgs_L.shape[0], 8)
        L_b = imgs_L[:b].detach().cpu()                 # [-1,1], Bx1xHxW
        ab_b = pred_ab[:b].detach().clamp(-1, 1).cpu()  # [-1,1], Bx2xHxW

        vis = []
        for k in range(b):
            L_255 = ((L_b[k, 0].numpy() + 1.0) * 0.5 * 255.0).astype(np.float32)     # H,W
            ab_255 = (ab_b[k].permute(1, 2, 0).numpy() * 127.0 + 128.0).astype(np.float32)  # H,W,2

            lab = np.zeros((cf.cellSize, cf.cellSize, 3), dtype=np.float32)
            lab[:, :, 0] = L_255
            lab[:, :, 1:3] = ab_255
            lab = np.clip(lab, 0, 255).astype(np.uint8)

            rgb = cv.cvtColor(lab, cv.COLOR_LAB2RGB)  # H,W,3 uint8
            t = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0  # [0,1]
            vis.append(t)

        grid = torch.stack(vis, dim=0)  # B,3,H,W
        torchvision.utils.save_image(grid, f"{log_dir}/_e_{i+1:03}.png")

last_path = f"{log_dir}/_ae_{cf.epochSize:03}.pth"
torch.save(model.state_dict(), last_path)

print(f"done {time.time()-s_tm:.01f}s")
print(f"best model: {best_path} (epoch={best_epoch}, loss={best_loss:.6f})")
print(f"last model: {last_path}")