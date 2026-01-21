import sys, os
sys.dont_write_bytecode = True
import random
import shutil
import pathlib

def dataset_split(master_root, train_ratio, all_data = False, g_colab=False): # データセットのランダム分割
    master_root = master_root.resolve()

    # ファイルのリストアップ
    IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    image_files = sorted([p for p in master_root.glob("**/*") if p.suffix.lower() in IMG_EXTS])
    LABEL_EXT = ".txt"
    
    if not image_files:
        sys.exit(1) # 画像ファイルがなければ終了

    random.seed(os.getpid()) # 実行ごとに異なるシードを設定
    random.shuffle(image_files) # シャッフル

    # 分割
    split_idx = int(len(image_files) * train_ratio)
    if all_data:
        train_files = image_files.copy()
    else:
        train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    print(f"Total: {len(image_files)}, Train: {len(train_files)}, Val: {len(val_files)}")
    
    if g_colab:
        split_root = pathlib.Path("/content/_dataset_yolo_dynamic_split") # Colab環境用のルートディレクトリ
    else:
        split_root = pathlib.Path(__file__).resolve() # スクリプトの場所を基準にする
        split_root = split_root.parent / "_dataset_yolo_dynamic_split" # シンボリックリンクを配置するルートディレクトリ
    
    if split_root.is_dir(): # 既存のディレクトリを削除し、再作成 (前のリンクをクリアするため)
        print(f"Cleaning previous split directory: {split_root}")
        shutil.rmtree(split_root)
        
    (split_root / "images" / "train").mkdir(parents=True)
    (split_root / "labels" / "train").mkdir(parents=True)
    (split_root / "images" / "val").mkdir(parents=True)
    (split_root / "labels" / "val").mkdir(parents=True)

    def create_links(file_list, type_name): # シンボリックリンクの作成
        img_target_dir = split_root / "images" / type_name
        label_target_dir = split_root / "labels" / type_name
        for img_path in file_list:
            # 画像ファイルのリンク作成
            img_link_path = img_target_dir / img_path.name
            os.symlink(img_path, img_link_path)
            
            # 対応するラベルファイルのリンク作成
            label_name = img_path.stem + LABEL_EXT
            label_path = master_root / label_name
            label_link_path = label_target_dir / label_name
            
            # ラベルファイルが存在する場合のみリンクを作成
            if label_path.exists():
                os.symlink(label_path, label_link_path)

    create_links(train_files, "train")
    create_links(val_files, "val")
    
    return split_root


if __name__ == "__main__":
    import pathlib
    root_dir = pathlib.Path(sys.argv[1])

    dataset_root_path = dataset_split(root_dir, 0.8)
