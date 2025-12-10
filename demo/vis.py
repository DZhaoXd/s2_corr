# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 20:51:59 2025

@author: 15642
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import json
import torch
import colorsys
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import detection_utils as utils


# ====================== 工具函数：读取词表与颜色生成 ======================
CITYSCAPES_COLOR_BY_NAME = {
    "road":           [128,  64, 128],
    "sidewalk":       [244,  35, 232],
    "building":       [ 70,  70,  70],
    "wall":           [102, 102, 156],
    "fence":          [190, 153, 153],
    "pole":           [153, 153, 153],
    "traffic light":  [250, 170,  30],
    "traffic sign":   [220, 220,   0],
    "vegetation":     [107, 142,  35],
    "terrain":        [152, 251, 152],
    "sky":            [ 70, 130, 180],
    "person":         [220,  20,  60],
    "rider":          [255,   0,   0],
    "car":            [  0,   0, 142],
    "truck":          [  0,   0,  70],
    "bus":            [  0,  60, 100],
    "train":          [  0,  80, 100],
    "motorcycle":     [  0,   0, 230],
    "bicycle":        [119,  11,  32],
}


def load_class_names_from_json(path):
    """
    读取 TEST_CLASS_JSON（list[str]）。
    例如:
    [
      "road", "sidewalk", "building", ...
    ]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list) and all(isinstance(x, str) for x in data), \
        "TEST_CLASS_JSON 必须是 list[str]"
    return data


def generate_distinct_colors(n, seed_shift=0.0):
    """
    生成 n 个分散颜色：黄金角间隔色相 + 交替饱和度/亮度，减少相邻ID相似。
    返回 [[r,g,b], ...], 0-255
    """
    phi = 0.61803398875
    out = []
    for k in range(n):
        h = (k * phi + seed_shift) % 1.0
        # 交替 s,v 以增强近邻差异
        s = 0.85 if k % 2 == 0 else 0.65
        v = 0.95 if (k // 2) % 2 == 0 else 0.78
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        out.append([int(r * 255), int(g * 255), int(b * 255)])
    return out  # list[[r,g,b]]


def build_palette_for_vocab(num_classes, ignore_index=255, base_colors=None):
    """
    返回：
      - stuff_colors: 长度为 num_classes 的 [[r,g,b], ...]（Visualizer 用）
      - palette_256:  长度恰为 256*3 的扁平列表（P 模式 PNG 用）
    规则：
      * 前面尽量用 Cityscapes 风格 base_colors
      * 不足部分用 generate_distinct_colors 填充
      * palette_256 的 ignore_index（默认 255）设为白色
    """
    if base_colors is None:
        base_colors = [
            [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
            [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
            [107, 142, 35], [152, 251, 152], [0, 130, 180], [220, 20, 60],
            [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
            [0, 80, 100], [0, 0, 230], [119, 11, 32],
        ]  # Cityscapes-19

    bs = len(base_colors)
    if num_classes <= bs:
        stuff_colors = base_colors[:num_classes]
    else:
        stuff_colors = base_colors + generate_distinct_colors(num_classes - bs, seed_shift=0.11)

    # --- 构造 256 色扁平调色板（索引PNG用） ---
    flat = []
    for c in stuff_colors:
        flat += c

    if num_classes < 256:
        extra = generate_distinct_colors(256 - num_classes, seed_shift=0.37)
        for c in extra:
            flat += c

    # 截断或填零到 256*3
    if len(flat) < 256 * 3:
        flat += [0] * (256 * 3 - len(flat))
    elif len(flat) > 256 * 3:
        flat = flat[:256 * 3]

    # ignore 颜色设为白色
    if isinstance(ignore_index, int) and 0 <= ignore_index < 256:
        flat[ignore_index * 3: ignore_index * 3 + 3] = [255, 255, 255]

    return stuff_colors, flat


def get_pred_mask(output):
    """
    将 Detectron2/CATSeg 的输出统一成 HxW 的 np.uint8 索引图。
    支持：
      - dict，含 'sem_seg'：C×H×W logits -> argmax；或 H×W 索引
      - 直接 H×W ndarray
    """
    if isinstance(output, dict):
        x = output.get("sem_seg", None)
        if x is None:
            raise ValueError("Output dict has no key 'sem_seg'.")
        if torch.is_tensor(x):
            x = x.detach().cpu()
        if isinstance(x, np.ndarray):
            pred = x
        else:
            if x.ndim == 3:  # C,H,W
                pred = x.argmax(dim=0).to(torch.uint8).cpu().numpy()
            elif x.ndim == 2:
                pred = x.to(torch.uint8).cpu().numpy()
            else:
                raise ValueError(f"Unexpected sem_seg shape: {x.shape}")
    else:
        pred = np.asarray(output, dtype=np.uint8)
    return pred



# ====================== CLI / 配置 ======================

def parse_args():
    parser = argparse.ArgumentParser("Semantic Segmentation Visualizer (train_net-style CLI)")
    parser.add_argument("--config", type=str, required=True, help="path to config file (train_net style)")
    parser.add_argument("--num-gpus", type=int, default=1, help="kept for CLI compatibility; not used here")
    parser.add_argument("--dist-url", type=str, default="auto", help="kept for CLI compatibility; not used here")
    parser.add_argument("--eval-only", action="store_true",
                        help="kept for CLI compatibility; this script always runs eval-style inference")
    parser.add_argument("opts", nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line 'KEY VALUE' pairs")
    return parser.parse_args()


def load_gt_mask(inp, ignore_label=255):
    """
    从 detectron2 的 dataset dict 读取 GT：
      - inp["sem_seg"] 可能是 tensor/ndarray
      - 或 inp["sem_seg_file_name"] 是路径（索引 PNG）
    返回 HxW 的 np.uint8（保留 ignore_label）
    """
    if "sem_seg" in inp:
        gt = inp["sem_seg"]
        if torch.is_tensor(gt):
            gt = gt.detach().cpu().numpy()
        else:
            gt = np.asarray(gt)
        gt = gt.astype(np.uint8)
        return gt

    if "sem_seg_file_name" in inp and os.path.isfile(inp["sem_seg_file_name"]):
        print('sem_seg_file_name')
        # 用 PIL 读索引 PNG；保持原值
        with Image.open(inp["sem_seg_file_name"]) as im:
            gt = np.array(im, dtype=np.uint8)
        return gt

    # 没有 GT（有些纯测试集可能不带标注）
    return None




# ====================== 调色板选择逻辑（多数据集支持） ======================

def build_palette_from_colors(colors, num_classes, ignore_index=255):
    """直接用已有 colors 构建 256 色调色板"""
    if len(colors) < num_classes:
        colors = list(colors) + generate_distinct_colors(num_classes - len(colors), seed_shift=0.23)
    stuff_colors = [list(map(int, c)) for c in colors[:num_classes]]

    flat = []
    for c in stuff_colors:
        flat += c
    if num_classes < 256:
        extra = generate_distinct_colors(256 - num_classes, seed_shift=0.47)
        for c in extra:
            flat += c
    if len(flat) < 256 * 3:
        flat += [0] * (256 * 3 - len(flat))
    elif len(flat) > 256 * 3:
        flat = flat[:256 * 3]

    # ignore = 白色
    if isinstance(ignore_index, int) and 0 <= ignore_index < 256:
        flat[ignore_index * 3: ignore_index * 3 + 3] = [255, 255, 255]

    return stuff_colors, flat


def get_dataset_palette(dataset_name, class_names, num_classes, ignore_label=255):
    """
    根据数据集名自动选择调色板方案。
      - roadwork：使用注册时元数据中的 stuff_colors
      - mapillary：对用户指定集合复用 Cityscapes 颜色；其余自动生成
      - 其他：默认黄金角 + HSV
    """
    meta = MetadataCatalog.get(dataset_name)
    name_l = dataset_name.lower()

    # -------- RoadWork --------
    if "roadwork" in name_l:
        print("[Palette] Using ROADWork official colors.")
        rw_colors = list(getattr(meta, "stuff_colors", []))
        return build_palette_from_colors(
            colors=rw_colors,
            num_classes=num_classes,
            ignore_index=ignore_label
        )

    # -------- Mapillary --------
    elif "mapillary" in name_l:
        print("[Palette] Using Cityscapes-mapped + HSV palette for Mapillary.")

        # 仅对你指定的这些类复用 Cityscapes 颜色（注意：按“类名精确匹配”）
        reusable_classes = {
              "road", "sidewalk", "building", "wall", "fence", "pole",
                "traffic light", "traffic sign", "vegetation", "terrain",
                "sky", "person", "rider", "car", "truck", "bus",
                "train", "motorcycle", "bicycle"
        }
        colors = []
        missing_idx = []

        for i, cname in enumerate(class_names):
            key = cname.lower()
            if key in reusable_classes and key in CITYSCAPES_COLOR_BY_NAME:
                colors.append(CITYSCAPES_COLOR_BY_NAME[key])
            else:
                colors.append(None)
                missing_idx.append(i)

        # 对缺失的类自动生成颜色
        if missing_idx:
            gen = generate_distinct_colors(len(missing_idx), seed_shift=0.33)
            for j, i in enumerate(missing_idx):
                colors[i] = gen[j]

        return build_palette_from_colors(colors, num_classes, ignore_label)


    # -------- Mapillary --------
    elif "anormaly" in name_l:
        print("[Palette] Using Cityscapes-mapped + HSV palette for Anormaly.")

        # 仅对你指定的这些类复用 Cityscapes 颜色（注意：按“类名精确匹配”）
        reusable_classes = {
              "road", "sidewalk", "building", "wall", "fence", "pole",
                "traffic light", "traffic sign", "vegetation", "terrain",
                "sky", "person", "rider", "car", "truck", "bus",
                "train", "motorcycle", "bicycle"
        }
        colors = []
        missing_idx = []

        for i, cname in enumerate(class_names):
            key = cname.lower()
            if key in reusable_classes and key in CITYSCAPES_COLOR_BY_NAME:
                colors.append(CITYSCAPES_COLOR_BY_NAME[key])
            else:
                colors.append(None)
                missing_idx.append(i)

        # 对缺失的类自动生成颜色
        if missing_idx:
            gen = generate_distinct_colors(len(missing_idx), seed_shift=0.33)
            for j, i in enumerate(missing_idx):
                colors[i] = gen[j]

        return build_palette_from_colors(colors, num_classes, ignore_label)

    # -------- Default --------
    else:
        print(f"[Palette] Using default generated palette for dataset: {dataset_name}")
        return build_palette_for_vocab(num_classes, ignore_index=ignore_label)


def _auto_font_size(h, w, base_ratio=0.06, min_px=40, max_px=1000):
    # 建议比例：约 4% 的短边，限定上下限
    fs = int(round(min(h, w) * base_ratio))
    return max(min_px, min(max_px, fs))


def _resize_for_vis(img_rgb, pred_idx, gt_idx=None, short_size=640):
    """
    将图像与标签按比例缩放，使短边=short_size，保持纵横比。
    彩色图使用双线性插值，索引图使用最近邻插值。
    """
    h, w = img_rgb.shape[:2]

    # 计算新的尺寸，短边为 short_size
    if h < w:
        new_h = short_size
        new_w = int(round(w * (short_size / h)))
    else:
        new_w = short_size
        new_h = int(round(h * (short_size / w)))

    img_v = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pred_v = cv2.resize(pred_idx, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    gt_v = None
    if gt_idx is not None:
        gt_v = cv2.resize(gt_idx, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    return img_v, pred_v, gt_v


def setup_cfg(args):
    cfg = get_cfg()
    # 可选：如果你的项目有扩展配置
    try:
        from cat_seg.config import add_cat_seg_config
        add_cat_seg_config(cfg)
    except Exception:
        pass

    cfg.merge_from_file(args.config)
    if args.opts:
        cfg.merge_from_list(args.opts)

    if not cfg.MODEL.DEVICE:
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    cfg.freeze()
    return cfg
# ====================== 主逻辑 ======================

def main():
    args = parse_args()
    cfg = setup_cfg(args)

    # 输出目录：使用 cfg.OUTPUT_DIR（与 train_net 一致）
    save_dir = cfg.OUTPUT_DIR if hasattr(cfg, "OUTPUT_DIR") and cfg.OUTPUT_DIR else "viz_out"
    os.makedirs(save_dir, exist_ok=True)

    # 数据集名称：取 cfg.DATASETS.TEST[0]
    test_sets = list(cfg.DATASETS.TEST) if hasattr(cfg.DATASETS, "TEST") else []
    if not test_sets:
        raise ValueError("cfg.DATASETS.TEST 为空，请通过命令行传入：DATASETS.TEST \"('your_dataset',)\"")
    dataset_name = test_sets[0]

    # 读取词表并构造颜色
    test_class_json = getattr(cfg.MODEL.SEM_SEG_HEAD, "TEST_CLASS_JSON", "")
    assert test_class_json, "请通过 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON 指定词表 JSON 路径（list[str])"
    class_names = load_class_names_from_json(test_class_json)
    num_classes = len(class_names)

    # 模型
    from detectron2.modeling import build_model
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()

    loader = build_detection_test_loader(cfg, dataset_name )

    from detectron2.evaluation import SemSegEvaluator
    evaluator = SemSegEvaluator(dataset_name, distributed=False)


    # 元数据 & 调色板

    meta = MetadataCatalog.get(dataset_name)
    ignore_label = getattr(meta, "ignore_label", 255)

    stuff_colors, PALETTE_256 = get_dataset_palette(
        dataset_name=dataset_name,
        class_names=class_names,
        num_classes=num_classes,
        ignore_label=ignore_label
    )
    if not hasattr(meta, "stuff_colors") or meta.stuff_colors is None:
        stuff_colors = [tuple(c) for c in stuff_colors]
        meta.set(stuff_classes=class_names, stuff_colors=stuff_colors)
    else:
        print(f"[Info] Using existing stuff_colors from MetadataCatalog for {dataset_name}.")

    # 可视化循环
    for idx, batch in enumerate(tqdm(loader, desc=f"vis:{dataset_name}")):
        with torch.no_grad():
            outputs = model(batch)

        for inp, out in zip(batch, outputs):
            # 原图（BGR->RGB）
            img = cv2.imread(inp["file_name"], cv2.IMREAD_COLOR)
            if img is None:
                # 读图失败时跳过
                continue
            img = img[:, :, ::-1]

            # 预测索引图
            pred = get_pred_mask(out)

            # --- 可选护栏：检测越界类别 ---
            max_id = int(pred.max()) if pred.size > 0 else -1
            if max_id >= num_classes and max_id != ignore_label:
                # 仅警告，不中断：有些模型可能包含背景/未定义类
                print(f"[Warn] pred max id {max_id} >= num_classes {num_classes}. "
                      f"Ensure your head / class json are aligned.")

            # --- 保存索引 PNG（P 模式 + 256 调色板） ---
            stem = os.path.splitext(os.path.basename(inp["file_name"]))[0]
            pred_img = Image.fromarray(pred.astype(np.uint8), mode="P")
            pred_img.putpalette(PALETTE_256)
            pred_png_path = os.path.join(save_dir, f"{stem}_index.png")
            # pred_img.save(pred_png_path)

            # --- 叠加可视化（透明混合，ignore 区域保持原图） ---
            # 构造 LUT：前 num_classes 行为 stuff_colors；其余为 0（不会被用到）
            lut = np.zeros((256, 3), dtype=np.uint8)
            lut[:len(stuff_colors)] = np.array(stuff_colors, dtype=np.uint8)
            color_mask = lut[pred]  # (H,W,3)


            gt_path = evaluator.input_file_to_gt_file[inp["file_name"]]  # 与eval一致的GT路径
            gt = np.array(Image.open(gt_path), dtype=np.uint8)
            # 若尺寸不匹配，调整到与原图一致（最近邻，保持索引）
            if gt.shape[:2] != img.shape[:2]:
                gt = cv2.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

            img_vis, pred_vis, gt_vis = _resize_for_vis(img, pred, gt)
            name_l = dataset_name.lower()
            if 'roadwork' in name_l:
                valid = (gt_vis != ignore_label)
            else:
                valid = (pred_vis != ignore_label)

            # 后续都用 *_vis 变量
            vis = Visualizer(img_vis, meta, instance_mode=ColorMode.IMAGE)
            vis._default_font_size = 32  # 固定一个好看的值即可
            vis = vis.draw_sem_seg(np.where(valid, pred_vis, 255))
            vis_img = vis.get_image()
            # 保存 vis_img
            out_path = os.path.join(save_dir, f"{stem}_overlay.jpg")
            Image.fromarray(vis_img).save(out_path)

            if gt_vis is not None:
                vis_gt = Visualizer(img_vis, meta, instance_mode=ColorMode.IMAGE)
                vis_gt._default_font_size = 32
                vis_gt = vis_gt.draw_sem_seg(np.where(gt_vis == ignore_label, 255, gt_vis))
                gt_img = vis_gt.get_image()
                # 保存 gt_img
                gt_out_path = os.path.join(save_dir, f"{stem}_gt.jpg")
                Image.fromarray(gt_img).save(gt_out_path)

                img_out_path = os.path.join(save_dir, f"{stem}.jpg")
                Image.fromarray(img_vis).save(img_out_path)


if __name__ == "__main__":
    main()
