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

colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255)] # クラスごとの色設定
fill_alpha = 0.4

proc_time = []
for f in range(len(fileList)):
    s_tm = time.time()
    image_path = fileList[f]

    img = cv.imread(image_path)
    
    res = model.predict(img, save = False, conf = 0.3, verbose = False)
    if res[0].masks is not None:
        masks = res[0].masks.data.cpu().numpy()
        classes = res[0].boxes.cls.cpu().numpy().astype(int)
        scores = res[0].boxes.conf.cpu().numpy()

        overlay = img.copy()
        for mask, cls_id, score in zip(masks, classes, scores):
            
            # マスク画像を元画像のサイズにリサイズして2値化 (0 or 255)
            mask_resized = cv.resize(mask, (img.shape[1], img.shape[0]))
            mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255

            contours, _ = cv.findContours(mask_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) # 輪郭抽出
            cv.drawContours(img, contours, -1, colors[cls_id], 2) # 輪郭線の描画

            mask_bool = mask_resized > 0.5
            overlay[mask_bool] = colors[cls_id] # マスク領域を色で塗りつぶす
        cv.addWeighted(overlay, fill_alpha, img, 1 - fill_alpha, 0, img)
    
    output_filename = f"{image_path.stem}_seg.png"
    output_img_path = output_dir / output_filename
    cv.imwrite(output_img_path, img)
    proc_time.append((time.time() - s_tm))

proc_time = np.array(proc_time)
print(np.mean(proc_time))