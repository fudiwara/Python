import sys
sys.dont_write_bytecode = True
import pathlib
import cv2 as cv
from ultralytics import YOLO

model = YOLO(sys.argv[1]) # モデルのパス
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

    bboxs = res[0].boxes.xyxy.detach().cpu().numpy() # xyxyの矩形情報
    scores = res[0].boxes.conf.detach().cpu().numpy() # スコアを取得

    for i in range(len(bboxs)):
        x0, y0, x1, y1 = bboxs[i]
        print(scores[i], x0, y0, x1, y1)
        p0 = (int(x0), int(y0))
        p1 = (int(x1), int(y1))
        cv.rectangle(img, p0, p1, (0, 255, 0), 2) # 検出結果の矩形描画

        score_text = f"{scores[i]:.2f}"
        cv.putText(img, score_text, (int(x0), int(y0) - 6), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    
    output_filename = f"{image_path.stem}_det.png"
    output_img_path = output_dir / output_filename
    cv.imwrite(output_img_path, img)
    proc_time.append((time.time() - s_tm))

proc_time = np.array(proc_time)
print(np.mean(proc_time))