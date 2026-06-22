import sys, time, os
sys.dont_write_bytecode = True
import statistics

import torch
from torch import nn
import torchvision
from torchvision import models

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
l1_loss = nn.L1Loss()

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=cf.epochSize,
    eta_min=1.0e-6
)

# perceptual loss (VGG16 features)
vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:16].to(DEVICE).eval()
for p in vgg.parameters():
    p.requires_grad = False

def perceptual_loss(pred, target):
    # pred/target: [-1,1] -> [0,1]
    p = (pred + 1.0) * 0.5
    t = (target + 1.0) * 0.5
    fp = vgg(p)
    ft = vgg(t)
    return nn.functional.l1_loss(fp, ft)

dataset = ld.load_datasets(dataset_path)
itr_size = max(1, cf.dataset_size // cf.batchSize)
s_tm = time.time()

with open(path_log, mode="w") as f:
    print("loss_total,loss_l1,loss_perc,lr,best_loss", file=f)

lambda_l1 = 100.0
lambda_perc = 10.0

# ===== best保存追加 =====
best_loss = float("inf")
best_epoch = -1
best_path = f"{log_dir}/best.pth"
# =======================

for i in range(cf.epochSize):
    model.train()
    log_total, log_l1, log_perc = [], [], []
    n_tm = time.time()

    for n, (real_target, imgs_src) in enumerate(dataset):
        real_target = real_target.to(DEVICE)  # 3ch
        imgs_src = imgs_src.to(DEVICE)        # 1ch

        fake_target = model(imgs_src)

        loss_l1 = l1_loss(fake_target, real_target)
        loss_perc = perceptual_loss(fake_target, real_target)
        loss = lambda_l1 * loss_l1 + lambda_perc * loss_perc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log_total.append(loss.item())
        log_l1.append(loss_l1.item())
        log_perc.append(loss_perc.item())

        print(
            f"\r {i+1:03}/{cf.epochSize:03} [{n+1:04}/{itr_size:04}] "
            f"L:{loss.item():.04f} L1:{loss_l1.item():.04f} P:{loss_perc.item():.04f}",
            end=""
        )

    scheduler.step()
    current_lr = optimizer.param_groups[0]["lr"]

    m_total = statistics.mean(log_total)
    m_l1 = statistics.mean(log_l1)
    m_perc = statistics.mean(log_perc)

    # ===== best更新 =====
    updated_best = ""
    if m_total < best_loss:
        best_loss = m_total
        best_epoch = i + 1
        torch.save(model.state_dict(), best_path)
        updated_best = " *best"
    # ===================

    print(
        f"\r {i+1:03}/{cf.epochSize:03} [{n+1:04}/{itr_size:04}] "
        f"L:{m_total:.04f} L1:{m_l1:.04f} P:{m_perc:.04f} "
        f"best:{best_loss:.04f}(e{best_epoch:03}) {time.time()-n_tm:.01f}s{updated_best}"
    )

    with open(path_log, mode="a") as f:
        print(f"{m_total},{m_l1},{m_perc},{current_lr},{best_loss}", file=f)

    # サンプル保存（fake / real）
    b = min(real_target.shape[0], 32)
    save_imgs = torch.cat([fake_target[:b], real_target[:b]], dim=0)
    torchvision.utils.save_image(
        save_imgs,
        f"{log_dir}/_e_{i+1:03}.png",
        value_range=(-1.0, 1.0),
        normalize=True
    )

# 最終モデルも保存
last_path = f"{log_dir}/_ae_{cf.epochSize:03}.pth"
torch.save(model.state_dict(), last_path)

print(f"done {time.time()-s_tm:.01f}s")
print(f"best model: {best_path} (epoch={best_epoch}, loss={best_loss:.6f})")
print(f"last model: {last_path}")