import sys
sys.dont_write_bytecode = True
import pathlib
import cv2 as cv
from ultralytics import YOLO

model = YOLO(sys.argv[1]) # モデルの読み込み
image_dir_path = pathlib.Path(sys.argv[2]) # 入力画像が入っているディレクトリのパス
output_dir = pathlib.Path(sys.argv[3]) # 画像を保存するフォルダ
output_dir.mkdir(parents = True, exist_ok = True) # ディレクトリ生成

exts = [".jpg", ".png", ".jpeg"] # 処理対象の拡張子
fileList = sorted([p for p in image_dir_path.glob("**/*") if p.suffix.lower()  in exts])

proc_time = []
for f in range(len(fileList)):
    s_tm = time.time()
    image_path = fileList[f]

    img = cv.imread(image_path)
    res = model.predict(img, save = False, conf = 0.3)

    polys = res[0].obb.xyxyxyxy.cpu().numpy() # ポリゴンの座標
    scores = res[0].obb.conf.detach().cpu().numpy() # スコアを取得

    for i in range(len(polys)):
        pts = polys[i].reshape(-1, 2).astype(int)
        cv.polylines(img, [pts], isClosed = True, color = (0, 255, 0), thickness = 2) # 検出結果のポリゴン描画
        print(scores[i], pts)

    output_filename = f"{image_path.stem}_det.png"
    output_img_path = output_dir / output_filename
    cv.imwrite(output_img_path, img)
    proc_time.append((time.time() - s_tm))

proc_time = np.array(proc_time)
print(np.mean(proc_time))