import sys
sys.dont_write_bytecode = True
import pathlib

from ultralytics import YOLO

id_str = sys.argv[1] # IDの文字列
dataset_root_dir = pathlib.Path(sys.argv[2]) # データセットのルートディレクトリ

output_dir = pathlib.Path("_log_" + id_str) # 出力ディレクトリ
output_dir.mkdir(exist_ok = True)

model = YOLO("yolo26n-cls.pt")
results = model.train(
    data = str(dataset_root_dir.resolve()),
    epochs = 20,
    imgsz = 96,
    batch = 64,
    workers = 2,
    project = str(output_dir.resolve().parent), # 親の出力ディレクトリ
    name = output_dir.name,
    exist_ok = True, # 上書き許可
    pretrained = True,
    optimizer = "auto",
    verbose = True,
)

print("Training finished.")
print(results)

