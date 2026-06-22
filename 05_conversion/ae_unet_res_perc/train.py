import sys, time, os
sys.dont_write_bytecode = True
import statistics

import torch
from torch import nn
import torchvision
import torchvision.models as models
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
# 余力があれば base_ch=48 推奨
model = cf.GeneratorAE(base_ch=32).to(DEVICE)
model = nn.DataParallel(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(0.9, 0.999), weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=cf.epochSize,
    eta_min=2e-5
)

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


perc_loss = PerceptualLoss().to(DEVICE)


def _f_lab_to_xyz(t):
    # t: >=0
    delta = 6.0 / 29.0
    return torch.where(t > delta, t ** 3, 3 * (delta ** 2) * (t - 4.0 / 29.0))


def _f_xyz_to_rgb(t):
    # sRGB gamma
    a = 0.055
    return torch.where(t <= 0.0031308, 12.92 * t, (1 + a) * torch.pow(torch.clamp(t, min=0.0), 1 / 2.4) - a)


def lab01_to_rgb01_torch(L_n, ab_n):
    """
    微分可能Lab->RGB変換（torch only）
    L_n:  [-1,1], Bx1xHxW
    ab_n: [-1,1], Bx2xHxW
    return: RGB [0,1], Bx3xHxW
    """
    # 学習時の正規化の逆変換（OpenCV由来の近似）
    # L_255 in [0,255], ab_255 in [0,255]
    L_255 = (L_n + 1.0) * 0.5 * 255.0
    a_255 = ab_n[:, 0:1] * 127.0 + 128.0
    b_255 = ab_n[:, 1:2] * 127.0 + 128.0

    # OpenCV Lab(8bit) -> CIE Lab 近似
    # L*: [0,100], a*: [-128,127], b*: [-128,127]
    L_star = L_255 * (100.0 / 255.0)
    a_star = a_255 - 128.0
    b_star = b_255 - 128.0

    # Lab -> XYZ (D65)
    fy = (L_star + 16.0) / 116.0
    fx = fy + (a_star / 500.0)
    fz = fy - (b_star / 200.0)

    Xn, Yn, Zn = 95.047, 100.000, 108.883
    X = Xn * _f_lab_to_xyz(fx)
    Y = Yn * _f_lab_to_xyz(fy)
    Z = Zn * _f_lab_to_xyz(fz)

    # XYZ (0..100) -> linear RGB
    x = X / 100.0
    y = Y / 100.0
    z = Z / 100.0

    r_lin =  3.2406 * x - 1.5372 * y - 0.4986 * z
    g_lin = -0.9689 * x + 1.8758 * y + 0.0415 * z
    b_lin =  0.0557 * x - 0.2040 * y + 1.0570 * z

    rgb_lin = torch.cat([r_lin, g_lin, b_lin], dim=1)
    rgb = _f_xyz_to_rgb(rgb_lin)
    rgb = torch.clamp(rgb, 0.0, 1.0)
    return rgb


dataset = ld.load_datasets(dataset_path)  # (real_ab, imgs_L)
itr_size = max(1, cf.dataset_size // cf.batchSize)
s_tm = time.time()

with open(path_log, mode="w") as f:
    print("loss_total,loss_ab,loss_perc,loss_sat,lambda_perc,lr,best_loss", file=f)

best_loss = float("inf")
best_epoch = -1
best_path = f"{log_dir}/_ae_best.pth"

for i in range(cf.epochSize):
    model.train()
    log_loss_total = []
    log_loss_ab = []
    log_loss_perc = []
    log_loss_sat = []
    n_tm = time.time()

    # 穏やかなウォームアップ
    # 1-10ep: 0.00, 11-40ep: 0.02->0.08, 41ep-: 0.08
    ep = i + 1
    if ep <= 10:
        lambda_perc = 0.00
    elif ep <= 40:
        lambda_perc = 0.02 + (0.08 - 0.02) * ((ep - 11) / (40 - 11))
    else:
        lambda_perc = 0.08

    lambda_sat = 0.01

    for n, (real_ab, imgs_L) in enumerate(dataset):
        real_ab = real_ab.to(DEVICE)  # [B,2,H,W]
        imgs_L  = imgs_L.to(DEVICE)   # [B,1,H,W]

        pred_ab = model(imgs_L)

        loss_ab = l1_loss(pred_ab, real_ab)

        # 微分可能RGB変換でperceptualを計算
        pred_rgb = lab01_to_rgb01_torch(imgs_L, pred_ab)
        real_rgb = lab01_to_rgb01_torch(imgs_L, real_ab)
        loss_perc = perc_loss(pred_rgb, real_rgb)

        sat_pred = torch.mean(torch.abs(pred_ab))
        sat_real = torch.mean(torch.abs(real_ab))
        loss_sat = torch.abs(sat_pred - sat_real)

        loss = loss_ab + lambda_perc * loss_perc + lambda_sat * loss_sat

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        log_loss_total.append(loss.item())
        log_loss_ab.append(loss_ab.item())
        log_loss_perc.append(loss_perc.item())
        log_loss_sat.append(loss_sat.item())

        print(
            f"\r {i+1:03}/{cf.epochSize:03} [{n+1:04}/{itr_size:04}] "
            f"tot:{loss.item():.04f} ab:{loss_ab.item():.04f} "
            f"perc:{loss_perc.item():.04f} sat:{loss_sat.item():.04f} wp:{lambda_perc:.03f}",
            end=""
        )

    scheduler.step()
    current_lr = optimizer.param_groups[0]["lr"]

    m_loss_total = statistics.mean(log_loss_total)
    m_loss_ab = statistics.mean(log_loss_ab)
    m_loss_perc = statistics.mean(log_loss_perc)
    m_loss_sat = statistics.mean(log_loss_sat)

    updated_best = ""
    if m_loss_total < best_loss:
        best_loss = m_loss_total
        best_epoch = i + 1
        torch.save(model.state_dict(), best_path)
        updated_best = " *best"

    print(
        f"\r {i+1:03}/{cf.epochSize:03} [{n+1:04}/{itr_size:04}] "
        f"tot:{m_loss_total:.04f} ab:{m_loss_ab:.04f} perc:{m_loss_perc:.04f} sat:{m_loss_sat:.04f} "
        f"wp:{lambda_perc:.03f} best:{best_loss:.04f}(e{best_epoch:03}) "
        f"lr:{current_lr:.7f} {time.time()-n_tm:.01f}s{updated_best}"
    )

    with open(path_log, mode="a") as f:
        print(f"{m_loss_total},{m_loss_ab},{m_loss_perc},{m_loss_sat},{lambda_perc},{current_lr},{best_loss}", file=f)

    # プレビュー保存（L + pred_ab -> RGB）
    with torch.no_grad():
        b = min(imgs_L.shape[0], 8)
        pred_rgb_b = lab01_to_rgb01_torch(imgs_L[:b], pred_ab[:b]).cpu()  # [0,1], Bx3xHxW
        torchvision.utils.save_image(pred_rgb_b, f"{log_dir}/_e_{i+1:03}.png")

last_path = f"{log_dir}/_ae_{cf.epochSize:03}.pth"
torch.save(model.state_dict(), last_path)

print(f"done {time.time()-s_tm:.01f}s")
print(f"best model: {best_path} (epoch={best_epoch}, loss={best_loss:.6f})")
print(f"last model: {last_path}")