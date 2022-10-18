# 事前にeffdetのインストールが必要
import sys, time, os
sys.dont_write_bytecode = True

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

import config as cf
import load_dataset as ld

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
if DEVICE == "cuda": torch.backends.cudnn.benchmark = True
id_str = sys.argv[1]
dataset_path_image = sys.argv[2]
dataset_path_annotation = sys.argv[3]
path_log = "_l_" + id_str + ".csv"
disp_score_t = ""
output_dir = "log" # 保存用ディレクトリ
if not os.path.exists(output_dir): os.mkdir(output_dir)

# pandasデータからDataset型のインスタンス作成
df_train, _, df_val = ld.read_csv_split_key(dataset_path_annotation, cf.splitRateTrain)
train_ds = ld.ImageFolderAnnotationRect(dataset_path_image, df_train)
val_ds = ld.ImageFolderAnnotationRect(dataset_path_image, df_val)

# 学習・検証データの準備
train_datasets = ld.EfficientDetDataset(train_ds, transforms=ld.get_train_transforms())
val_datasets = ld.EfficientDetDataset(val_ds)
train_loader = DataLoader(train_datasets, batch_size=cf.batchSize, shuffle=True, num_workers = os.cpu_count(), pin_memory=True, drop_last=True, collate_fn=ld.collate_fn_cstm,)
val_loader = DataLoader(val_datasets, batch_size=cf.batchSize, shuffle=True, num_workers = os.cpu_count(), pin_memory=True, drop_last=True, collate_fn=ld.collate_fn_cstm,)
print(f"train size: {len(train_datasets)}, val size: {len(val_datasets)}")
print(f"train batch: {len(train_loader)}, val batch: {len(val_loader)}")

# モデル＋損失関数、最適化関数、収束率の定義
model = cf.create_model(cf.numClasses, cf.imageSize, architecture=cf.modelArchitecture)
optimizer = optim.Adam(model.parameters(), lr=cf.learningRate)

def Train_Eval(model,optimizer,data_loader,device,epoch,max_epoch,is_val=False):
    total_loss = 0.0
    global disp_score_t
    model.eval() if is_val else model.train()
    for n, (data, annot) in enumerate(data_loader): # バッチ毎にデータ読み込み
        if is_val == False: optimizer.zero_grad()

        if is_val:
            with torch.no_grad():
                output = model(data, annot)
        else:
            output = model(data, annot)
        
        loss = output['loss']
        total_loss += loss.item()

        if is_val == False:
            loss.backward()
            optimizer.step()
        
        if is_val == False:
            disp_score_t = "{:03} / {:03} [ {:04} / {:04} ] l: {:.05f}".format(epoch+1, max_epoch, n + 1, len(data_loader), total_loss/(n+1))
            print("\r {}".format(disp_score_t), end = "")
        else: 
            print("\r {} l: {:.05f}".format(disp_score_t, total_loss/(n+1)), end = "")

    return total_loss/(n+1)

best_loss = None
with open(path_log, mode = "w") as f: f.write("")
s_tm = time.time()

for epoch in range(cf.epochSize):
    n_tm = time.time()
    train_loss = Train_Eval(model,optimizer,train_loader,DEVICE,epoch,cf.epochSize) 
    val_loss = Train_Eval(model,optimizer,val_loader,DEVICE,epoch,cf.epochSize,is_val=True)
    print(" %.0fs" % (time.time() - n_tm))

    if best_loss is None or val_loss < best_loss: # 評価での誤差がよき感じでモデルを保存
        best_loss = val_loss
        torch.save(model.state_dict(), output_dir + "/_m_" + id_str + "_" + str(epoch).zfill(3) + ".pth")

    # 学習の状況をCSVに保存
    with open(path_log, mode = "a") as f: f.write("%f,%f\n" % (train_loss, val_loss))

# モデルの保存
torch.save(model.state_dict(), output_dir + "/_m_" + id_str + "_" + str(cf.epochSize).zfill(3) + ".pth")
print("done %.0fs" % (time.time() - s_tm))
