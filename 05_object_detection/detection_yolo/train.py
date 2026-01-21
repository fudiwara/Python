import sys
sys.dont_write_bytecode = True
import yaml
import pathlib
from ultralytics import YOLO

import setup_dataset_det as ds

id_str = sys.argv[1] # IDの文字列
dataset_root_dir = pathlib.Path(sys.argv[2]) # データセットのルートディレクトリ

dataset_path = ds.dataset_split(dataset_root_dir, 0.8, all_data = True, g_colab=True) # YOLO用のデータセット構築

output_dir = pathlib.Path("_log_" + id_str) # 出力ディレクトリ
output_dir.mkdir(exist_ok = True)

data_dict = {
    "path": str(dataset_path),
    "train": "images/train",
    "val": "images/val",
    "nc": 1, # クラス数
    "names": ["hoge"] # クラス名のリスト
}
yaml_path = pathlib.Path("_buf.yaml") # 一時的なYAMLファイル
with open(yaml_path, mode = "w", encoding = "utf-8") as f:
    yaml.safe_dump(data_dict, f, allow_unicode = True, default_flow_style = False)

model = YOLO("yolo11s.pt") 
results = model.train(
    data = yaml_path, # データセット設定ファイルへのパス
    epochs = 20, # エポック数
    imgsz = 512, # 画像サイズ
    batch = -1, # CUDAの60%になるように
    degrees = 5.0, # 回転の角度
    rect = True, # 矩形学習を有効化
    name = ".", # output_dir直下に保存するため"."にする
    project = output_dir, # 親の出力ディレクトリ
    exist_ok = True # 上書き許可
)
yaml_path.unlink() # 一時的なYAMLファイルを削除
