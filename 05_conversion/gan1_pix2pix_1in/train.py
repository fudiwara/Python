import sys, time, pathlib
sys.dont_write_bytecode = True
import statistics

import torch
import torchvision

import load_dataset as ld
import config as cf

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
torch.backends.cudnn.benchmark = True

id_str = sys.argv[1]
dataset_path = pathlib.Path(sys.argv[2])
path_log = "_l_" + id_str + ".csv"
log_dir = pathlib.Path("_log_" + id_str)
log_dir.mkdir(exist_ok=True, parents=True)

model_G, model_D = cf.Generator().to(DEVICE), cf.Discriminator().to(DEVICE)

params_G = torch.optim.Adam(model_G.parameters(), lr=cf.lr_g, betas=(cf.beta1, cf.beta2))
params_D = torch.optim.Adam(model_D.parameters(), lr=cf.lr_d, betas=(cf.beta1, cf.beta2))

bce_loss = torch.nn.BCEWithLogitsLoss()
mae_loss = torch.nn.L1Loss()

scaler_G = torch.amp.GradScaler("cuda", enabled=(cf.use_amp and DEVICE == "cuda"))
scaler_D = torch.amp.GradScaler("cuda", enabled=(cf.use_amp and DEVICE == "cuda"))

dataset = ld.load_datasets(dataset_path)
itr_size = max(1, cf.dataset_size // cf.batchSize)
print(cf.dataset_size, itr_size)
s_tm = time.time()

with open(path_log, mode="w") as f:
    print("gl_mean,gl_bce,gl_l1,dl", file=f)

best_score = 1e18
best_epoch = -1

for i in range(cf.epochSize):
    model_G.train()
    model_D.train()

    log_loss_G_sum, log_loss_G_bce, log_loss_G_mae, log_loss_D = [], [], [], []
    n_tm = time.time()

    for n, (real_target, imgs_src) in enumerate(dataset):
        real_target = real_target.to(DEVICE, non_blocking=True)
        imgs_src = imgs_src.to(DEVICE, non_blocking=True)

        # 生成器の更新
        params_G.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(cf.use_amp and DEVICE == "cuda")):
            fake_target = model_G(imgs_src)
            out_fake_for_g = model_D(torch.cat([fake_target, imgs_src], dim=1))
            ones_g = torch.ones_like(out_fake_for_g)
            loss_G_bce = bce_loss(out_fake_for_g, ones_g)
            loss_G_mae = cf.lambda_l1 * mae_loss(fake_target, real_target)
            loss_G_sum = loss_G_bce + loss_G_mae

        scaler_G.scale(loss_G_sum).backward()
        scaler_G.step(params_G)
        scaler_G.update()

        # 識別器の更新
        update_d = (n % cf.d_update_interval == 0)
        if update_d:
            params_D.zero_grad(set_to_none=True)
            fake_det = fake_target.detach()

            in_real = torch.cat([real_target, imgs_src], dim=1)
            in_fake = torch.cat([fake_det, imgs_src], dim=1)

            if cf.use_input_noise_for_d: # 学習安定化のため識別器の入力にノイズを加える
                in_real = in_real + cf.input_noise_std * torch.randn_like(in_real)
                in_fake = in_fake + cf.input_noise_std * torch.randn_like(in_fake)

            with torch.amp.autocast("cuda", enabled=(cf.use_amp and DEVICE == "cuda")):
                out_real = model_D(in_real)
                out_fake = model_D(in_fake)

                ones_d = torch.full_like(out_real, cf.real_label_smooth)
                zeros_d = torch.zeros_like(out_fake)

                loss_D_real = bce_loss(out_real, ones_d)
                loss_D_fake = bce_loss(out_fake, zeros_d)
                loss_D = 0.5 * (loss_D_real + loss_D_fake)

            scaler_D.scale(loss_D).backward()
            scaler_D.step(params_D)
            scaler_D.update()
        else:
            loss_D = torch.zeros((), device=DEVICE)

        log_loss_G_bce.append(loss_G_bce.item())
        log_loss_G_mae.append(loss_G_mae.item())
        log_loss_G_sum.append(loss_G_sum.item())
        log_loss_D.append(loss_D.item())

        print(f"\r {i + 1:03}/{cf.epochSize:03} [{n + 1:04}/{itr_size:04}] GL_bce:{loss_G_bce.item():.4f} L1:{loss_G_mae.item():.4f} DL:{loss_D.item():.4f}", end = "")

    gl_mean = statistics.mean(log_loss_G_sum)
    gl_bce = statistics.mean(log_loss_G_bce)
    gl_l1 = statistics.mean(log_loss_G_mae)
    dl = statistics.mean(log_loss_D)

    print(f"\r {i + 1:03}/{cf.epochSize:03} [{n + 1:04}/{itr_size:04}] GL_bce:{gl_bce:.4f} L1:{gl_l1:.4f} DL:{dl:.4f} {time.time() - n_tm:.1f}s")

    with open(path_log, mode="a") as f:
        print(f"{gl_mean},{gl_bce},{gl_l1},{dl}", file=f)

    vis_n = min(real_target.size(0), 32)
    fake_vis = fake_target.detach()
    buf_save_imgs = torch.cat([fake_vis[:vis_n], real_target[:vis_n]], dim=0)
    torchvision.utils.save_image(buf_save_imgs, log_dir / f"_e_{i + 1:03}.png", value_range=(-1.0, 1.0), normalize=True)

    # L1による簡易ベスト保存
    score = gl_l1 + 0.2 * gl_bce
    if score < best_score:
        best_score = score
        best_epoch = i + 1
        torch.save(model_G.state_dict(), log_dir / "best.pth")

torch.save(model_G.state_dict(), log_dir / f"_gen_{cf.epochSize:03}.pth")
torch.save(model_D.state_dict(), log_dir / f"_dis_{cf.epochSize:03}.pth")
print(f"best epoch: {best_epoch}, score: {best_score:.4f}")
print(f"done {time.time() - s_tm:.1f}s")