import sys, os, time,pathlib
sys.dont_write_bytecode = True

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import config as cf
import load_dataset_dirs as ld

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
model_path = sys.argv[1] # モデルのパス
dataset_path = pathlib.Path(sys.argv[2]) # テスト用の画像が入ったディレクトリのパス

# モデルの定義と読み込みおよび評価用のモードにセットする
model = cf.build_model("eval").to(DEVICE)
model.load_state_dict(torch.load(model_path, map_location = DEVICE, weights_only = False))
model.eval()

# データの読み込み (バッチサイズは適宜変更する)
s_tm = time.time()
test_data = ld.ImageFolder_directory(dataset_path, cf.transforms_eval)
print(test_data.class_to_idx)
bs = cf.batchSize
# bs = int(bs * 0.5) + 1 # 必要メモリ量に応じた調整 (場合によっては1以下をかける)
test_loader = DataLoader(test_data, batch_size = bs, num_workers = os.cpu_count())

label_list, pred_list = [], []
for i, (data, label) in enumerate(test_loader):
    data = data.to(DEVICE)
    label = label.numpy().tolist()
    with torch.no_grad(): # 推定のために勾配計算の無効化モードで
        outputs = model(data)
    pred = torch.argmax(outputs, axis = 1).cpu().numpy().tolist()
    label_list += label
    pred_list += pred

    print(f"\r dataset loading: {i + 1} / {len(test_loader)}", end = "", flush = True)
print()

print(accuracy_score(label_list, pred_list)) # 正解率
print(confusion_matrix(label_list, pred_list)) # 混同行列
print(classification_report(label_list, pred_list)) # 各種評価指標
print("done %.0fs" % (time.time() - s_tm))