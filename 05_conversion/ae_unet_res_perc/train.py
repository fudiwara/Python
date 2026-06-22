import sys, time, os
sys.dont_write_bytecode = True
import statistics

import torch
from torch import nn
import torchvision
import torchvision.models as models
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

# 基本損失
l1_loss = nn.L1Loss()


class PerceptualLoss(nn.Module):
    """
    VGG16特徴を使ったPerceptual Loss。
    入力は[0,1]のRGB画像(B,3,H,W)を想定。
    """
    def __init__(self, layers=(8, 15), layer_weights=(1.0, 1.0)):
        super().__init__()
        weights = models.VGG16_Weights.IMAGENET1K_V1
        vgg_features = models.vgg16(weights=weights).features.eval()

        self.slices = nn.ModuleList()
        prev = 0
        for l in layers:
            self.slices.append(nn.Sequential(*[vgg_features[i] for i in range(prev, l)]))
            prev = l

        for p in self.parameters():
            p.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.layer_weights = layer_weights

    def _norm(self, x):
        return (x - self.mean) / self.std

    def forward(self, x, y):
        x = self._norm(x)
        y = self._norm(y)

        loss = 0.0
        for w, block in zip(self.layer_weights, self.slices):
            x = block(x)
            y = block(y)
            loss = loss + w * torch.mean(torch.abs(x - y))
        return loss


# Perceptual Loss（RGB再構成画像に適用）
perc_loss = PerceptualLoss().to(DEVICE)
lambda_perc = 0.05


def lab01_to_rgb01(L_n, ab_n):
    """
    L_n: [-1,1], Bx1xHxW
    ab_n: [-1,1], Bx2xHxW
    return: [0,1], Bx3xHxW (torch)

    OpenCV Labスケール(L,a,bとも0..255系)で合成し、RGBへ変換。
    """
    bsz = L_n.shape[0]
    out = []
    for k in range(bsz):
        L_255 = ((L_n[k, 0].detach().cpu().numpy() + 1.0) * 0.5 * 255.0).astype(np.float32)  # H,W
        ab_255 = (ab_n[k].detach().cpu().permute(1, 2, 0).numpy() * 127.0 + 128.0).astype(np.float32)  # H,W,2

        h, w = L_255.shape
        lab = np.zeros((h, w, 3), dtype=np.float32)
        lab[:, :, 0] = L_255
        lab[:, :, 1:3] = ab_255
        lab = np.clip(lab, 0, 255).astype(np.uint8)

        rgb = cv.cvtColor(lab, cv.COLOR_LAB2RGB)  # H,W,3 uint8
        t = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0  # [0,1]
        out.append(t)

    return torch.stack(out, dim=0).to(L_n.device)


dataset = ld.load_datasets(dataset_path)  # (real_ab, imgs_L)
itr_size = max(1, cf.dataset_size // cf.batchSize)
s_tm = time.time()

with open(path_log, mode="w") as f:
    print("loss_total,loss_ab,loss_perc,lr,best_loss", file=f)

best_loss = float("inf")
best_epoch = -1
best_path = f"{log_dir}/_ae_best.pth"

for i in range(cf.epochSize):
    model.train()
    log_loss_total = []
    log_loss_ab = []
    log_loss_perc = []
    n_tm = time.time()

    for n, (real_ab, imgs_L) in enumerate(dataset):
        real_ab = real_ab.to(DEVICE)  # [B,2,H,W]
        imgs_L  = imgs_L.to(DEVICE)   # [B,1,H,W]

        pred_ab = model(imgs_L)

        loss_ab = l1_loss(pred_ab, real_ab)

        # perceptualはRGB空間で計算
        pred_rgb = lab01_to_rgb01(imgs_L, pred_ab)
        real_rgb = lab01_to_rgb01(imgs_L, real_ab)
        loss_perc = perc_loss(pred_rgb, real_rgb)

        loss = loss_ab + lambda_perc * loss_perc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log_loss_total.append(loss.item())
        log_loss_ab.append(loss_ab.item())
        log_loss_perc.append(loss_perc.item())

        print(
            f"\r {i+1:03}/{cf.epochSize:03} [{n+1:04}/{itr_size:04}] "
            f"tot:{loss.item():.04f} ab:{loss_ab.item():.04f} perc:{loss_perc.item():.04f}",
            end=""
        )

    scheduler.step()
    current_lr = optimizer.param_groups[0]["lr"]

    m_loss_total = statistics.mean(log_loss_total)
    m_loss_ab = statistics.mean(log_loss_ab)
    m_loss_perc = statistics.mean(log_loss_perc)

    updated_best = ""
    if m_loss_total < best_loss:
        best_loss = m_loss_total
        best_epoch = i + 1
        torch.save(model.state_dict(), best_path)
        updated_best = " *best"

    print(
        f"\r {i+1:03}/{cf.epochSize:03} [{n+1:04}/{itr_size:04}] "
        f"tot:{m_loss_total:.04f} ab:{m_loss_ab:.04f} perc:{m_loss_perc:.04f} "
        f"best:{best_loss:.04f}(e{best_epoch:03}) "
        f"lr:{current_lr:.7f} {time.time()-n_tm:.01f}s{updated_best}"
    )

    with open(path_log, mode="a") as f:
        print(f"{m_loss_total},{m_loss_ab},{m_loss_perc},{current_lr},{best_loss}", file=f)

    # プレビュー保存（L + pred_ab -> RGB）
    with torch.no_grad():
        b = min(imgs_L.shape[0], 8)
        L_b = imgs_L[:b].detach().cpu()                 # [-1,1], Bx1xHxW
        ab_b = pred_ab[:b].detach().clamp(-1, 1).cpu()  # [-1,1], Bx2xHxW

        vis = []
        for k in range(b):
            L_255 = ((L_b[k, 0].numpy() + 1.0) * 0.5 * 255.0).astype(np.float32)     # H,W
            ab_255 = (ab_b[k].permute(1, 2, 0).numpy() * 127.0 + 128.0).astype(np.float32)  # H,W,2

            h, w = L_255.shape
            lab = np.zeros((h, w, 3), dtype=np.float32)
            lab[:, :, 0] = L_255
            lab[:, :, 1:3] = ab_255
            lab = np.clip(lab, 0, 255).astype(np.uint8)

            rgb = cv.cvtColor(lab, cv.COLOR_LAB2RGB)  # H,W,3 uint8
            t = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0  # [0,1]
            vis.append(t)

        grid = torch.stack(vis, dim=0)  # B,3