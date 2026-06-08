import sys, math
sys.dont_write_bytecode = True
import torch
import torchvision

# 繰り返す回数
epochSize = 50

# 学習時のバッチのサイズ
batchSize = 3

# 勾配の累積ステップ数
accumulation_steps = 2

# カテゴリの総数 (背景の0を含めた合計)
numClasses = 2

# FasterRCNNの画像のリサイズ後の大きさ
min_size = 1024 # 最小辺
max_size = 1280 # 最大辺

# 検出の閾値
thDetection = 0.6

# データセットを学習用と評価用に分割する際の割合
splitRateTrain = 0.8

def build_model(sw_train_eval):
    if sw_train_eval == "train":
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights = weights, min_size = min_size, max_size = max_size)
    else: # 推論用
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights = None, min_size = min_size, max_size = max_size)
   
    in_features = model.roi_heads.box_predictor.cls_score.in_features # 事前学習済みのモデルの入力特徴量数を取得
    
    # ヘッドを新しいクラス数用に置き換え (FastRCNNPredictorを利用)
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, numClasses)

    return model

def estimate_batch_size_for_GPU(model, target_mem_rate = 0.8, DEVICE = "cuda"):
    model.train()
    total_memory = torch.cuda.get_device_properties(DEVICE).total_memory # GPUの総メモリ量
    torch.cuda.empty_cache() # キャッシュをクリア
    torch.cuda.reset_peak_memory_stats() # ピークメモリ使用量のリセット

    print(f"Total GPU memory: {total_memory / 1024**3:.2f} GB")

    base_memory = torch.cuda.memory_allocated(DEVICE) # モデルを動かす前のメモリ使用量
    print(f"Initial GPU memory: {base_memory / 1024**3:.2f} GB")

    dummy_input = torch.randn(1, 3, max_size, max_size).to(DEVICE)
    dummy_target = [{
        "boxes": torch.tensor([[0, 0, 100, 100]], dtype=torch.float32).to(DEVICE),
        "labels": torch.tensor([1], dtype=torch.int64).to(DEVICE)
    }]

    loss_dict = model(dummy_input, dummy_target)
    losses = sum(loss for loss in loss_dict.values()) # 損失の合計を求める: Backward用
    losses.backward()

    peak_memory = torch.cuda.max_memory_allocated(DEVICE) # モデルを動かした後のピークメモリ使用量
    memory_per_sample = peak_memory - base_memory # 1サンプルあたりのメモリ使用量
    print(f"Model memory: {memory_per_sample / 1024**3:.2f} GB")

    optimizer_margin = memory_per_sample * 2 # 最適化関数用のバッファ SGD: 1 〜 Adam系: 2 (調整する)
    available_mem = total_memory * target_mem_rate - (base_memory + optimizer_margin) # 実際におおよそ利用できるメモリ量
    print(f"Available GPU memory: {available_mem / 1024**3:.2f} GB")

    estimated_batch_size = available_mem / memory_per_sample
    valid_batch_size = int(estimated_batch_size)
    print(f"{estimated_batch_size} -> {valid_batch_size}")

    return valid_batch_size # 推定されたバッチサイズ

if __name__ == "__main__":
    from torchinfo import summary
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = build_model("train").to(DEVICE)
    print(mdl)
    summary(mdl, (batchSize, 3, max_size, max_size))
    est_batchSize = estimate_batch_size_for_GPU(mdl, target_mem_rate = 0.8, DEVICE = DEVICE)
    print(f"batch size: {est_batchSize}")
    