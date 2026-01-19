import sys, os, time, pathlib
sys.dont_write_bytecode = True
import torch
from torch.utils.data import DataLoader
from pyt_det.engine import train_one_epoch, evaluate
import config as cf
import load_dataset_coco_anno as ld

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
id_str = sys.argv[1]
img_dir_path = sys.argv[2]
annot_json_path = sys.argv[3]
path_log = "_l_" + id_str + ".csv" # loss推移の記録ファイル
output_dir = pathlib.Path("_log_" + id_str) # 保存用ディレクトリ
output_dir.mkdir(parents = True, exist_ok = True) # ディレクトリ生成

# 作成したカスタム・データセット (1つのものを分割するのではなく、同じものを2つ作ってそれぞれ使う)
train_dataset = ld.loadImagesCocoJson(img_dir_path, annot_json_path, ld.get_transform(train=True))
val_dataset = ld.loadImagesCocoJson(img_dir_path, annot_json_path, ld.get_transform(train=False))

# データセットを訓練セットとテストセットに分割
# torch.manual_seed(1)
indices = torch.randperm(len(train_dataset)).tolist()
train_data_size = int(cf.splitRateTrain * len(indices))
train_dataset = torch.utils.data.Subset(train_dataset, indices[:train_data_size])
val_dataset = torch.utils.data.Subset(val_dataset, indices[train_data_size:])
print(len(indices), len(train_dataset), len(val_dataset))

# 訓練データと評価データのデータロード用オブジェクトを用意
train_loader = DataLoader(train_dataset, batch_size=cf.batchSize, shuffle=True, num_workers=int(os.cpu_count() / 2), collate_fn=ld.collate_fn)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=int(os.cpu_count() / 2), collate_fn=ld.collate_fn)

# モデル、損失関数、最適化関数、収束率の定義
model = cf.build_model("train").to(DEVICE)
params = [p for p in model.parameters() if p.requires_grad]
# optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
# optimizer = torch.optim.SGD(params, lr=0.0005, momentum=0.9, weight_decay=0.0005)
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1) # 3エポックごとに学習率が1/10
optimizer = torch.optim.Adam(model.parameters(), lr=0.00002, betas=(0.5, 0.999))

with open(path_log, mode = "w") as f: print(f"loss,f1v", file = f)
s_tm = time.time()
for epoch in range(cf.epochSize):
    loss = train_one_epoch(model, optimizer, train_loader, DEVICE, epoch, print_freq=1) # 学習
    # lr_scheduler.step() # 学習率の更新
    f1v = evaluate(model, val_loader, device=DEVICE) # テストデータセットの評価
    torch.save(model.state_dict(), f"{output_dir}/_m_{id_str}_{epoch + 1:03}.pth") # モデルの保存

    # 学習の状況をCSVに保存
    with open(path_log, mode = "a") as f: print(f"{loss},{f1v}", file = f)
print("done %.0fs" % (time.time() - s_tm))