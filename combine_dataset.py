# com d0 l0 d1 l1 ...
# python combine_dataset.py /Users/tfuji/work/hep/dataset_comb/ _hep_train_list.txt /Users/tfuji/work/hep/worker1 /Users/tfuji/work/hep/worker1/_yolo_train_list.txt /Users/tfuji/work/hep/worker2_sq /Users/tfuji/work/hep/worker2_sq/_yolo_train_list.txt /Users/tfuji/work/hep/worker3_1 /Users/tfuji/work/hep/worker3_1/_yolo_train_list.txt /Users/tfuji/work/hep/worker3_2sq /Users/tfuji/work/hep/worker3_2sq/_yolo_train_list.txt /Users/tfuji/work/hep/worker3_3 /Users/tfuji/work/hep/worker3_3/_yolo_train_list.txt

import sys
sys.dont_write_bytecode = True
import pathlib
import shutil

ext = [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"]
output_dir = pathlib.Path(sys.argv[1]) # コピー先のディレクトリ
list_combine = sys.argv[2] # 結合後のリストファイル名
fw = open(list_combine, mode = "w")
if(not output_dir.exists()): output_dir.mkdir() # ディレクトリ生成
print(len(sys.argv))
filne_name_list = []
for n in range(2, int(len(sys.argv) / 2)):
    img_dir_path = pathlib.Path(sys.argv[n * 2 - 1])
    annot_file_name = pathlib.Path(sys.argv[n * 2])
    print(n)
    print(img_dir_path)
    print(annot_file_name)
    if img_dir_path.is_dir() == False:
        print(f"[{n}] ディレクトリを指定してください: {str(img_dir_path)}")
        exit()
    lines = []
    with open(annot_file_name, "r") as f: # 各リストを開く
        for line in f:
            l = line.split(" ")
            if 1 < len(l): # フォーマットチェック
                if l[0] not in filne_name_list: # ファイル名重複チェック
                    filne_name_list.append(l[0])
                    fw.write(line) # フォーマットが正しい行を残す
                    shutil.copy(img_dir_path / l[0], output_dir)

fw.close()