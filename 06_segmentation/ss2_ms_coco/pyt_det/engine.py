import math
import os, sys
import time
import io
from contextlib import redirect_stdout

import torch
import torchvision.models.detection.mask_rcnn
import pyt_det.utils as U
from pyt_det.coco_eval import CocoEvaluator
from pyt_det.coco_utils import get_coco_api_from_dataset

import config as cf
def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    s_tm = time.time()
    model.train()
    metric_logger = U.MetricLogger(delimiter=" ")
    metric_logger.add_meter("lr", U.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"[{epoch + 1} / {cf.epochSize}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header, True):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # print(targets[0]["image_id"])
        # with torch.cuda.amp.autocast(enabled=scaler is not None):
        # with torch.amp.autocast("cuda", enabled=scaler is not None):
        with torch.autocast(device_type=device, enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = U.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    el_tm = time.time() - s_tm
    print(f"t_loss: {losses.item():.04f}")

    # return metric_logger
    return losses.item()


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = U.MetricLogger(delimiter=" ")
    header = "Eval:"

    f = io.StringIO() # 準備段階の標準出力をバッファに逃がす
    with redirect_stdout(f):
        coco = get_coco_api_from_dataset(data_loader.dataset)

        if "info" not in coco.dataset:
            coco.dataset["info"] = []
        if "licenses" not in coco.dataset:
            coco.dataset["licenses"] = []

        iou_types = _get_iou_types(model)
        coco_evaluator = CocoEvaluator(coco, iou_types) # modelの構成から iou_types (["box", "segm"] など) を取得

    for images, targets in metric_logger.log_every(data_loader, 100, header): # 推論ループ
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        coco_evaluator.update(res)

    with redirect_stdout(f): # 統計の集計の標準出力をバッファへ
        metric_logger.synchronize_between_processes()
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    f1_scores = {} # IoUタイプごとのF1スコアを格納する辞書
    
    for iou_type in iou_types:
        # coco_eval[iou_type].stats に AP/AR 等の数値が入っている
        stats = coco_evaluator.coco_eval[iou_type].stats
        ap = stats[0]  # AP @[ IoU=0.50:0.95 ]
        ar = stats[8]  # AR @[ IoU=0.50:0.95 | maxDets=100 ]
        
        if ap + ar == 0:
            f1 = 0.0
        else:
            f1 = 2 * ap * ar / (ap + ar)
        
        f1_scores[iou_type] = f1
        
        print(f"{iou_type:4s} AP: {ap:.3f}, AR: {ar:.3f}, F1: {f1:.4f}")

    # 返り値としては、bboxのF1値を優先的に返す（学習ログ用）
    main_f1 = f1_scores.get("bbox", list(f1_scores.values())[0] if f1_scores else 0.0)

    torch.set_num_threads(n_threads)
    return main_f1
