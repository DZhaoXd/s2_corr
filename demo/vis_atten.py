#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from pathlib import Path
import argparse
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.engine import default_argument_parser
from detectron2.modeling import build_model
from detectron2.utils.visualizer import Visualizer, ColorMode
from tqdm import tqdm

# ====================== 实用函数 ======================

def load_class_names_from_json(json_path: str) -> List[str]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 支持 {"classes":[...]} 或 直接 list
    if isinstance(data, dict) and "classes" in data:
        return list(map(str, data["classes"]))
    elif isinstance(data, list):
        return list(map(str, data))
    else:
        raise ValueError(f"Unrecognized class json format: {json_path}")

def _to_uint8_bgr(img_rgb: np.ndarray) -> np.ndarray:
    """img_rgb: [H,W,3], float(0..1 / 0..255) or uint8"""
    if img_rgb.dtype != np.uint8:
        x = img_rgb
        if x.max() <= 1.0 + 1e-6:
            x = (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        else:
            x = np.clip(x, 0.0, 255.0).astype(np.uint8)
        img_rgb = x
    return img_rgb[..., ::-1].copy()  # to BGR

def _normalize_map(m: np.ndarray, method='percentile', eps=1e-6, p_low=2.0, p_high=98.0):
    # m: [H,W] float
    if method == 'percentile':
        lo = np.percentile(m, p_low)
        hi = np.percentile(m, p_high)
        if hi - lo < eps:
            return np.zeros_like(m, dtype=np.float32)
        m = (m - lo) / (hi - lo)
    else:
        mn, mx = float(m.min()), float(m.max())
        if mx - mn < eps:
            return np.zeros_like(m, dtype=np.float32)
        m = (m - mn) / (mx - mn)
    return np.clip(m, 0.0, 1.0)

def _overlay_heatmap_on_image(base_img, heatmap, alpha=0.5, cmap=cv2.COLORMAP_JET):
    """
    base_img: HxWx{3|4} 或 HxW，任意 dtype / 通道
    heatmap:  HxW 或 HxWx1/3，任意 dtype
    返回:     HxWx3 uint8 的 BGR 叠加图
    """
    # ---- 1) 规范 base 到 BGR 3通道 uint8 ----
    base = base_img
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    elif base.ndim == 3 and base.shape[2] == 4:
        base = cv2.cvtColor(base, cv2.COLOR_BGRA2BGR)
    elif base.ndim == 3 and base.shape[2] == 3:
        pass
    else:
        raise ValueError(f"[overlay] Unexpected base shape: {base.shape}")
    if base.dtype != np.uint8:
        base = np.clip(base, 0, 255).astype(np.uint8)

    H, W = base.shape[:2]

    # ---- 2) 规范 heatmap 到 HxW uint8 ----
    hm = heatmap
    if hm.ndim == 3:
        hm = hm[..., 0]  # 若是 HxWx1/3，取第一通道
    if hm.shape[:2] != (H, W):
        hm = cv2.resize(hm, (W, H), interpolation=cv2.INTER_LINEAR)

    hm = hm.astype(np.float32)
    hmin, hmax = np.nanmin(hm), np.nanmax(hm)
    if hmax > hmin:
        hm = (hm - hmin) / (hmax - hmin)
    else:
        hm = np.zeros_like(hm, dtype=np.float32)
    hm_u8 = (hm * 255).astype(np.uint8)

    # ---- 3) 着色到 BGR ----
    if isinstance(cmap, int):
        hm_color = cv2.applyColorMap(hm_u8, cmap)
    else:
        try:
            cv2_cmap = getattr(cv2, f"COLORMAP_{str(cmap).upper()}")
            hm_color = cv2.applyColorMap(hm_u8, cv2_cmap)
        except Exception:
            m = cm.get_cmap(cmap if isinstance(cmap, str) else "jet")
            hm_rgb = (m(hm)[:, :, :3] * 255).astype(np.uint8)  # RGB
            hm_color = cv2.cvtColor(hm_rgb, cv2.COLOR_RGB2BGR)

    # ---- 4) alpha 合法化并融合 ----
    alpha = float(alpha)
    alpha = 0.0 if alpha < 0 else (1.0 if alpha > 1 else alpha)
    out = cv2.addWeighted(hm_color, alpha, base, 1.0 - alpha, 0.0)
    return out


# def visualize_corr_and_logit(images_rgb: torch.Tensor,
#                              corr: torch.Tensor,      # [B,1,T,Hc,Wc]
#                              logit: torch.Tensor,     # [B,T,Hl,Wl]
#                              out_dir: str,
#                              class_names: Optional[List[str]] = None,
#                              alpha: float = 0.5,
#                              upsample_mode: str = 'bilinear',
#                              normalize: str = 'percentile',
#                              topk: Optional[int] = None,
#                              cmap=cv2.COLORMAP_JET,
#                              gap_px: int = 24,            # ← 新增：图与图之间的留白宽度
#                              gap_color: int = 245):       # ← 新增：留白颜色(0~255，灰白更通用)
#     """
#     对每张图、每个类，保存：
#       sample_xxx/{cls}/corr_overlay.jpg
#       sample_xxx/{cls}/logit_overlay.jpg
#       sample_xxx/{cls}/triplet.jpg   # 三图中间有留白
#     """
#     Path(out_dir).mkdir(parents=True, exist_ok=True)
#     B, _, H_img, W_img = images_rgb.shape
#
#     # 上采样到原图大小
#     corr_up = F.interpolate(corr.squeeze(1), size=(H_img, W_img),
#                             mode=upsample_mode, align_corners=False)  # [B,T,H,W]
#     logit_up = F.interpolate(logit,         size=(H_img, W_img),
#                              mode=upsample_mode, align_corners=False) # [B,T,H,W]
#
#     images_np = images_rgb.detach().cpu().float().permute(0, 2, 3, 1).numpy()  # [B,H,W,3]
#     corr_np   = corr_up.detach().cpu().float().numpy()
#     logit_np  = logit_up.detach().cpu().float().numpy()
#
#     Tb = corr_np.shape[1]
#     for b in range(B):
#         if topk is not None and topk < Tb:
#             mean_act = logit_np[b].reshape(Tb, -1).mean(axis=1)
#             sel = np.argsort(-mean_act)[:topk]
#         else:
#             sel = np.arange(Tb)
#
#         base_dir = Path(out_dir) / f"sample_{b:03d}"
#         base_dir.mkdir(parents=True, exist_ok=True)
#         img_bgr = _to_uint8_bgr(images_np[b])
#
#         # 预生成留白条（与图同高、gap_px 宽、三通道）
#         H, W = img_bgr.shape[:2]
#         pad = np.full((H, gap_px, 3), gap_color, dtype=np.uint8)
#
#         for t in sel:
#             cname = class_names[t] if (class_names and t < len(class_names)) else f"class_{t:02d}"
#             cls_dir = base_dir / f"{cname}"
#             cls_dir.mkdir(parents=True, exist_ok=True)
#
#             hm_corr  = _normalize_map(corr_np[b, t],  method=normalize)
#             hm_logit = _normalize_map(logit_np[b, t], method=normalize)
#
#             corr_overlay  = _overlay_heatmap_on_image(images_np[b], hm_corr,  alpha=alpha, cmap=cmap)
#             logit_overlay = _overlay_heatmap_on_image(images_np[b], hm_logit, alpha=alpha, cmap=cmap)
#
#             # 单图保存
#             # cv2.imwrite(str(cls_dir / "corr_overlay.jpg"),  corr_overlay)
#             # cv2.imwrite(str(cls_dir / "logit_overlay.jpg"), logit_overlay)
#
#             # 拼接三图：原图 | 留白 | corr | 留白 | logit
#             triplet = np.concatenate([img_bgr, pad, corr_overlay, pad, logit_overlay], axis=1)
#             cv2.imwrite(str(cls_dir / "triplet.jpg"), triplet)


def _resize_to_short_edge(img: np.ndarray, short=480):
    """等比例缩放，使短边=short"""
    H, W = img.shape[:2]
    if min(H, W) == short:
        return img
    if H < W:
        new_H, new_W = short, int(W * short / H)
    else:
        new_H, new_W = int(H * short / W), short
    return cv2.resize(img, (new_W, new_H), interpolation=cv2.INTER_LINEAR)

def visualize_corr_and_logit(
        images_rgb: torch.Tensor,        # [B,3,H,W], RGB, 0..1 或 0..255
        corr: torch.Tensor,              # [B,1,T,Hc,Wc]
        logit: torch.Tensor,             # [B,T,Hl,Wl]
        out_dir: str,
        class_names: Optional[List[str]] = None,
        alpha: float = 0.5,
        upsample_mode: str = 'bilinear',
        topk: Optional[int] = None,
        cmap: int = cv2.COLORMAP_JET,
        gap_px: int = 24,
        gap_color: int = 245,
        normalize: str = 'percentile',
        temperature: float = 0.9,
        short_edge: int = 640
    ):
    """
    三联图：原图 | corr(独立归一) | logit_softmax(跨类竞争)
    统一在短边缩放后的尺寸上进行上采样与叠图，确保尺寸/通道/数值范围一致。
    注意：为避免 OpenCV 的 BGR 假色问题，叠图底图统一使用 BGR。

    依赖的工具函数（你已有）：
      - _resize_to_short_edge(img_hw3, short_edge) -> resized_img_hw3
      - _normalize_map(map_hw, method='percentile'|...) -> 0..1 heatmap
      - _to_uint8_bgr(img_hw3_rgb_or_uint8) -> uint8 BGR
      - _overlay_heatmap_on_image(base_bgr_uint8, heatmap01, alpha=..., cmap=...) -> uint8 BGR
    """
    from pathlib import Path
    import numpy as np, cv2, torch
    import torch.nn.functional as F

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    B = images_rgb.shape[0]

    # 0) 将原图 tensor -> numpy，并按短边 resize；此处仍保持 RGB（数值范围 0..1 或 0..255 都可）
    images_np_list = []
    for b in range(B):
        # [3,H,W] -> [H,W,3]
        img = images_rgb[b].detach().cpu().float().permute(1, 2, 0).numpy()
        img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)  # 清理异常
        img = _resize_to_short_edge(img, short_edge)               # 仍是 RGB
        images_np_list.append(img)

    H_img, W_img = images_np_list[0].shape[:2]

    # 1) 上采样到与可视化底图一致的尺寸
    #    corr: [B,1,T,Hc,Wc] -> [B,T,Hc,Wc] -> upsample [B,T,H,W]
    corr_up  = F.interpolate(corr.squeeze(1), size=(H_img, W_img),
                             mode=upsample_mode, align_corners=False)
    #    logit: [B,T,Hl,Wl] -> upsample [B,T,H,W]
    logit_up = F.interpolate(logit,           size=(H_img, W_img),
                             mode=upsample_mode, align_corners=False)

    # 2) 跨类 softmax（温度缩放）
    t_eps = max(temperature, 1e-6)
    logit_sm = torch.softmax(logit_up / t_eps, dim=1)  # [B,T,H,W]

    # 3) 转 numpy
    images_np   = np.stack(images_np_list, axis=0)             # [B,H,W,3], RGB
    corr_np     = corr_up.detach().cpu().float().numpy()       # [B,T,H,W]
    logit_sm_np = logit_sm.detach().cpu().float().numpy()      # [B,T,H,W]

    Tb = corr_np.shape[1]
    for b in range(B):
        # 选类（top-k 平均概率）
        if topk is not None and topk < Tb:
            mean_prob = logit_sm_np[b].reshape(Tb, -1).mean(axis=1)
            sel = np.argsort(-mean_prob)[:topk]
        else:
            sel = np.arange(Tb)

        # 输出目录
        base_dir = Path(out_dir) / f"sample_{b:03d}"
        base_dir.mkdir(parents=True, exist_ok=True)

        # —— 关键修复点 ——：
        # 叠图底图改成 BGR（OpenCV 期望 BGR，避免把 RGB 当 BGR 造成“红蓝互换”）
        base_img_rgb = images_np[b]                # RGB, float(0..1/255)
        img_bgr      = _to_uint8_bgr(base_img_rgb) # uint8 BGR，可直接喂给 OpenCV

        H, W = img_bgr.shape[:2]
        pad = np.full((H, gap_px, 3), gap_color, dtype=np.uint8)

        # 保存一份原图面板（BGR）用于拼接
        # cv2.imwrite(str(base_dir / "image_bgr.jpg"), img_bgr)  # 如需单独保存可解注释

        for t in sel:
            cname = class_names[t] if (class_names and t < len(class_names)) else f"class_{t:02d}"
            cls_dir = base_dir / f"{cname}"
            cls_dir.mkdir(parents=True, exist_ok=True)

            # corr：独立归一 -> 0..1
            hm_corr = _normalize_map(corr_np[b, t], method=normalize)

            # 叠图：底图使用 BGR（已修复）
            corr_vis_bgr  = _overlay_heatmap_on_image(img_bgr, hm_corr,                   alpha=alpha, cmap=cmap)
            logit_vis_bgr = _overlay_heatmap_on_image(img_bgr, logit_sm_np[b, t],        alpha=alpha, cmap=cmap)

            # 单图保存（BGR）
            cv2.imwrite(str(cls_dir / "corr_overlay.jpg"),  corr_vis_bgr)
            cv2.imwrite(str(cls_dir / "logit_overlay.jpg"), logit_vis_bgr)

            # 三联拼接：原图 | corr | logit
            panel = np.concatenate([img_bgr, pad, corr_vis_bgr, pad, logit_vis_bgr], axis=1)
            cv2.imwrite(str(cls_dir / "triplet_comp.jpg"), panel)






# ====================== 稳健版捕获（默认用这个） ======================

import types

# ====================== 更稳的 monkey-patch：优先抓 correlation() ======================
import types
import torch

def register_corr_logit_monkeypatch(model):
    """
    直接改写子模块的 forward / 成员方法：
      1) 优先在聚合器的 correlation() 上抓 corr（输出 [B,1,T,H,W]）
      2) 若 correlation 不存在，回退到 corr_embed.forward 的输入
      3) 在聚合器 head.forward 的输出抓 logit（[B,T,H,W]）
      4) 再不行，回退到顶层 outputs['sem_seg']
    返回 (cache, undo_fns)
    """
    cache = {"corr": None, "logit": None}
    undo = []

    # --------- 1) 先找带 correlation() 的聚合器 ----------
    agg = None
    corr_hooked = False
    for m in model.modules():
        if hasattr(m, "correlation") and callable(getattr(m, "correlation")):
            agg = m
            orig_corr = m.correlation
            def corr_patched(self, *args, **kwargs):
                out = orig_corr(*args, **kwargs)
                # out 期望为 Tensor [B,1,T,H,W] 或可从 tuple/dict 中取到
                try:
                    if isinstance(out, torch.Tensor) and out.dim() == 5 and out.size(1) == 1:
                        cache["corr"] = out.detach()
                    elif isinstance(out, (list, tuple)):
                        for o in out:
                            if isinstance(o, torch.Tensor) and o.dim() == 5 and o.size(1) == 1:
                                cache["corr"] = o.detach(); break
                    elif isinstance(out, dict):
                        for v in out.values():
                            if isinstance(v, torch.Tensor) and v.dim() == 5 and v.size(1) == 1:
                                cache["corr"] = v.detach(); break
                except Exception:
                    pass
                return out
            m.correlation = types.MethodType(corr_patched, m)
            undo.append(lambda: setattr(m, "correlation", orig_corr))
            corr_hooked = True
            print("[hook] corr will be captured at correlation() output")
            break

    # --------- 2) 若没找到 correlation()，退回 corr_embed.forward 的输入 ----------
    if not corr_hooked:
        for m in model.modules():
            if hasattr(m, "corr_embed") and hasattr(m.corr_embed, "forward"):
                ce = m.corr_embed
                orig_ce_forward = ce.forward
                def ce_forward_patched(self, x, *args, **kwargs):
                    # x 期望为 [B,1,T,H,W]
                    try:
                        if isinstance(x, torch.Tensor) and x.dim() == 5 and x.size(1) == 1:
                            cache["corr"] = x.detach()
                    except Exception:
                        pass
                    return orig_ce_forward(x, *args, **kwargs)
                ce.forward = types.MethodType(ce_forward_patched, ce)
                undo.append(lambda: setattr(ce, "forward", orig_ce_forward))
                corr_hooked = True
                print("[hook] corr will be captured at corr_embed.forward input")
                break

        if not corr_hooked:
            print("[warn] neither correlation() nor corr_embed found — corr may remain None")

    # --------- 3) 在聚合器 head.forward 的输出抓 logit ----------
    logit_hooked = False
    # 优先找与 agg 同层的 head；若 agg 为空则全局找一个叫 head 的
    head_candidates = []
    if agg is not None and hasattr(agg, "head") and hasattr(agg.head, "forward"):
        head_candidates = [agg.head]
    else:
        for m in model.modules():
            if hasattr(m, "forward") and (m.__class__.__name__.lower().endswith("head") or hasattr(m, "weight")):
                # 粗略挑一个可能的 head；真正判定在 wrapper 内做
                head_candidates.append(m)

    for head in head_candidates:
        orig_head_forward = head.forward
        def head_forward_patched(self, x, *args, **kwargs):
            y = orig_head_forward(x, *args, **kwargs)
            try:
                if isinstance(y, torch.Tensor) and y.dim() == 4:
                    cache["logit"] = y.detach()
            except Exception:
                pass
            return y
        head.forward = types.MethodType(head_forward_patched, head)
        undo.append(lambda h=head, f=orig_head_forward: setattr(h, "forward", f))
        logit_hooked = True
        print(f"[hook] logit will be captured at {head.__class__.__name__}.forward output")
        break

    # --------- 4) 顶层回退：outputs['sem_seg'] ----------
    if not logit_hooked:
        orig_model_forward = model.forward
        def model_forward_patched(*args, **kwargs):
            y = orig_model_forward(*args, **kwargs)
            if isinstance(y, (list, tuple)) and len(y) > 0 and isinstance(y[0], dict) and "sem_seg" in y[0]:
                cache["logit"] = y[0]["sem_seg"].detach()
            elif isinstance(y, dict) and "sem_seg" in y:
                cache["logit"] = y["sem_seg"].detach()
            return y
        model.forward = types.MethodType(model_forward_patched, model)
        undo.append(lambda: setattr(model, "forward", orig_model_forward))
        print("[hook] logit fallback: captured from model outputs['sem_seg']")

    return cache, undo

def cache_clear(cache: dict):
    cache["corr"] = None
    cache["logit"] = None


def cache_clear(cache: dict):
    cache["corr"] = None
    cache["logit"] = None

# ====================== 你的原始 hook 方案（备用） ======================

# ====================== 更稳的 monkey-patch：优先抓 correlation() ======================
import types
import torch

def register_corr_and_logit_hooks(model):
    """
    直接改写子模块的 forward / 成员方法：
      1) 优先在聚合器的 correlation() 上抓 corr（输出 [B,1,T,H,W]）
      2) 若 correlation 不存在，回退到 corr_embed.forward 的输入
      3) 在聚合器 head.forward 的输出抓 logit（[B,T,H,W]）
      4) 再不行，回退到顶层 outputs['sem_seg']
    返回 (cache, undo_fns)
    """
    cache = {"corr": None, "logit": None}
    undo = []

    # --------- 1) 先找带 correlation() 的聚合器 ----------
    agg = None
    corr_hooked = False
    for m in model.modules():
        if hasattr(m, "correlation") and callable(getattr(m, "correlation")):
            agg = m
            orig_corr = m.correlation
            def corr_patched(self, *args, **kwargs):
                out = orig_corr(*args, **kwargs)
                # out 期望为 Tensor [B,1,T,H,W] 或可从 tuple/dict 中取到
                try:
                    if isinstance(out, torch.Tensor) and out.dim() == 5 and out.size(1) == 1:
                        cache["corr"] = out.detach()
                    elif isinstance(out, (list, tuple)):
                        for o in out:
                            if isinstance(o, torch.Tensor) and o.dim() == 5 and o.size(1) == 1:
                                cache["corr"] = o.detach(); break
                    elif isinstance(out, dict):
                        for v in out.values():
                            if isinstance(v, torch.Tensor) and v.dim() == 5 and v.size(1) == 1:
                                cache["corr"] = v.detach(); break
                except Exception:
                    pass
                return out
            m.correlation = types.MethodType(corr_patched, m)
            undo.append(lambda: setattr(m, "correlation", orig_corr))
            corr_hooked = True
            print("[hook] corr will be captured at correlation() output")
            break

    # --------- 2) 若没找到 correlation()，退回 corr_embed.forward 的输入 ----------
    if not corr_hooked:
        for m in model.modules():
            if hasattr(m, "corr_embed") and hasattr(m.corr_embed, "forward"):
                ce = m.corr_embed
                orig_ce_forward = ce.forward
                def ce_forward_patched(self, x, *args, **kwargs):
                    # x 期望为 [B,1,T,H,W]
                    try:
                        if isinstance(x, torch.Tensor) and x.dim() == 5 and x.size(1) == 1:
                            cache["corr"] = x.detach()
                    except Exception:
                        pass
                    return orig_ce_forward(x, *args, **kwargs)
                ce.forward = types.MethodType(ce_forward_patched, ce)
                undo.append(lambda: setattr(ce, "forward", orig_ce_forward))
                corr_hooked = True
                print("[hook] corr will be captured at corr_embed.forward input")
                break

        if not corr_hooked:
            print("[warn] neither correlation() nor corr_embed found — corr may remain None")

    # --------- 3) 在聚合器 head.forward 的输出抓 logit ----------
    logit_hooked = False
    # 优先找与 agg 同层的 head；若 agg 为空则全局找一个叫 head 的
    head_candidates = []
    if agg is not None and hasattr(agg, "head") and hasattr(agg.head, "forward"):
        head_candidates = [agg.head]
    else:
        for m in model.modules():
            if hasattr(m, "forward") and (m.__class__.__name__.lower().endswith("head") or hasattr(m, "weight")):
                # 粗略挑一个可能的 head；真正判定在 wrapper 内做
                head_candidates.append(m)

    for head in head_candidates:
        orig_head_forward = head.forward
        def head_forward_patched(self, x, *args, **kwargs):
            y = orig_head_forward(x, *args, **kwargs)
            try:
                if isinstance(y, torch.Tensor) and y.dim() == 4:
                    cache["logit"] = y.detach()
            except Exception:
                pass
            return y
        head.forward = types.MethodType(head_forward_patched, head)
        undo.append(lambda h=head, f=orig_head_forward: setattr(h, "forward", f))
        logit_hooked = True
        print(f"[hook] logit will be captured at {head.__class__.__name__}.forward output")
        break

    # --------- 4) 顶层回退：outputs['sem_seg'] ----------
    if not logit_hooked:
        orig_model_forward = model.forward
        def model_forward_patched(*args, **kwargs):
            y = orig_model_forward(*args, **kwargs)
            if isinstance(y, (list, tuple)) and len(y) > 0 and isinstance(y[0], dict) and "sem_seg" in y[0]:
                cache["logit"] = y[0]["sem_seg"].detach()
            elif isinstance(y, dict) and "sem_seg" in y:
                cache["logit"] = y["sem_seg"].detach()
            return y
        model.forward = types.MethodType(model_forward_patched, model)
        undo.append(lambda: setattr(model, "forward", orig_model_forward))
        print("[hook] logit fallback: captured from model outputs['sem_seg']")

    return cache, undo

def cache_clear(cache: dict):
    cache["corr"] = None
    cache["logit"] = None


# ====================== 配置与主逻辑 ======================

def parse_args():
    parser = default_argument_parser()
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument("--output", default="", help="save dir (default=cfg.OUTPUT_DIR/viz_attn)")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--normalize", type=str, default="percentile", choices=["percentile", "minmax"])
    parser.add_argument("--topk", type=int, default=10, help="only save top-k classes per image by mean logit")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--use_monkeypatch", action="store_true",
                        help="use robust monkey-patch hooks (recommended)")
    parser.add_argument("--opts", default=[], nargs=argparse.REMAINDER)
    return parser.parse_args()

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

def main():
    args = parse_args()
    cfg = setup_cfg(args)

    # 输出目录
    save_root = args.output.strip() or (
        os.path.join(cfg.OUTPUT_DIR, "viz_attn") if hasattr(cfg, "OUTPUT_DIR") else "viz_attn"
    )
    os.makedirs(save_root, exist_ok=True)

    # 数据集
    test_sets = list(cfg.DATASETS.TEST) if hasattr(cfg.DATASETS, "TEST") else []
    if not test_sets:
        raise ValueError("cfg.DATASETS.TEST 为空，请通过命令行传入：DATASETS.TEST \"('your_dataset',)\"")
    dataset_name = test_sets[0]

    # 词表
    test_class_json = getattr(cfg.MODEL.SEM_SEG_HEAD, "TEST_CLASS_JSON", "")
    assert test_class_json, "请通过 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON 指定词表 JSON 路径（list[str])"
    class_names = load_class_names_from_json(test_class_json)
    num_classes = len(class_names)

    # 模型
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval().to(cfg.MODEL.DEVICE)

    # loader
    loader = build_detection_test_loader(cfg, dataset_name)

    # 选择 hook 方式（推荐 monkey-patch）
    if args.use_monkeypatch:
        cache, undo_fns = register_corr_logit_monkeypatch(model)
        handles = None
    else:
        cache, handles = register_corr_and_logit_hooks(model)
        undo_fns = None

    # 可视化循环
    count = 0
    for idx, batch in enumerate(tqdm(loader, desc=f"viz:{dataset_name}")):
        if count >= args.max_samples:
            break

        # 读原图 & 组装 images_rgb（用原图尺寸可视化）
        images_rgb = []
        for inp in batch:
            img_bgr = cv2.imread(inp["file_name"], cv2.IMREAD_COLOR)
            if img_bgr is None:
                images_rgb.append(None)
                continue
            images_rgb.append(img_bgr[:, :, ::-1].copy())  # BGR->RGB

        # 每个 batch 前清空 cache（非常重要）
        cache["corr"]  = None
        cache["logit"] = None

        # 前向
        with torch.no_grad():
            _ = model(batch)  # Detectron2 接收 list[dict]

        corr  = cache["corr"]   # [B,1,T,hc,wc]
        logit = cache["logit"]  # [B,T,hl,wl] 或 [B,K,hl,wl]
        # --- 形状对齐：把 (B*T,1,h,w) 的 logit 还原成 [B,T,h,w] ---
        # 来自 corr 的维度：corr: [B, 1, T, hc, wc]
        Bcorr = int(corr.shape[0])
        Tcorr = int(corr.shape[2])

        if isinstance(logit, torch.Tensor) and logit.dim() == 4:
            # 情况1：标准 [B, T, h, w]（或 [B, num_classes, h, w]）-> 无需处理
            if logit.shape[0] == Bcorr and logit.shape[1] in (Tcorr,):
                pass
            # 情况2：抓在 head.forward，通常为 (B*T, 1, h, w) -> 还原
            elif logit.shape[0] == Bcorr * Tcorr and logit.shape[1] == 1:
                logit = logit.view(Bcorr, Tcorr, logit.shape[-2], logit.shape[-1])
                print(f"[fix] reshaped logit from (B*T,1,h,w) to [B={Bcorr}, T={Tcorr}, h, w]")
            # 情况3：其它形状，给出提示但继续可视化（会按索引命名）
            else:
                print(f"[warn] unexpected logit shape {tuple(logit.shape)}; "
                      f"expected [B,{Tcorr},h,w] or [B*T,1,h,w].")


        if (corr is None) or (logit is None):
            print("[warn] corr or logit not captured in this batch, skip.")
            continue

        # 打包 images_rgb 为张量（若尺寸不一致，这里统一为第一个有效尺寸）
        valid_imgs = [im for im in images_rgb if im is not None]
        if len(valid_imgs) == 0:
            continue
        H_img, W_img = valid_imgs[0].shape[:2]
        imgs_stack, keep_idx = [], []
        for b, im in enumerate(images_rgb):
            if im is None:
                continue
            if im.shape[:2] != (H_img, W_img):
                im = cv2.resize(im, (W_img, H_img), interpolation=cv2.INTER_LINEAR)
            imgs_stack.append(im.transpose(2,0,1))  # to CHW
            keep_idx.append(b)

        if len(keep_idx) != corr.shape[0]:
            # 若 loader 有图像尺寸不一导致过滤，此处对 corr/logit 同样筛选
            corr  = corr[keep_idx]
            logit = logit[keep_idx]

        images_rgb_tensor = torch.from_numpy(np.stack(imgs_stack, 0)).float().to(corr.device)  # [B,3,H,W]

        # 若 logit 的通道数 != num_classes，仅提示；仍照样保存（以 idx 命名）
        if logit.shape[1] != num_classes:
            print(f"[warn] logit channels ({logit.shape[1]}) != num_classes ({num_classes}), 将使用索引命名。")

        # 输出目录：按首张图名分文件夹
        stem0 = Path(batch[0]["file_name"]).stem
        out_dir = os.path.join(save_root, stem0)
        visualize_corr_and_logit(
            images_rgb=images_rgb_tensor,     # [B,3,H,W]
            corr=corr,                        # [B,1,T,hc,wc]
            logit=logit,                      # [B,T,hl,wl]
            out_dir=out_dir,
            class_names=class_names if logit.shape[1] == num_classes else None,
            alpha=args.alpha,
            upsample_mode='bilinear',
            # normalize=args.normalize,
            topk=args.topk,
            cmap=cv2.COLORMAP_JET
        )

        count += len(keep_idx)

    # 清理 hooks / 撤销 monkey-patch
    if handles is not None:
        for h in handles:
            h.remove()
    if undo_fns is not None:
        for f in undo_fns:
            try: f()
            except: pass

    print(f"[done] visualizations saved under: {save_root}")

if __name__ == "__main__":
    main()
