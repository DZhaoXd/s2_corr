# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple
import math

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList

from einops import rearrange




@META_ARCH_REGISTRY.register()
class S2Corr(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        size_divisibility: int,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        clip_pixel_mean: Tuple[float],
        clip_pixel_std: Tuple[float],
        train_class_json: str,
        test_class_json: str,
        sliding_window: bool,
        sw_kernel: int = 384,
        sw_overlap: float = 0.333,
        sw_out_res: tuple = (640, 640),
        enable_high_res: bool = True,
        clip_finetune: str = "prompt",
        backbone_multiplier: float = 1.0,
        clip_pretrained: str = "ViT-B/16",
    ):
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        if size_divisibility < 0:
            size_divisibility = self.backbone.size_divisibility if self.backbone is not None else 32
        self.size_divisibility = size_divisibility

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_mean", torch.Tensor(clip_pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_std", torch.Tensor(clip_pixel_std).view(-1, 1, 1), False)

        self.train_class_json = train_class_json
        self.test_class_json = test_class_json
        self.clip_finetune = clip_finetune

        # ===== sliding_window =====
        self.sliding_window = sliding_window
        self.sw_kernel = sw_kernel
        self.sw_overlap = sw_overlap
        self.sw_out_res = sw_out_res

        self.clip_resolution = (512, 512) if clip_pretrained in ["ViT-B/16", "EVA02-CLIP-B-16"] else (448, 448)

        # 通道与预训练一致
        self.proj_dim = 768 if clip_pretrained in ["ViT-B/16", "EVA02-CLIP-B-16"] else 1024
        self.upsample1 = nn.ConvTranspose2d(self.proj_dim, 256, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(self.proj_dim, 128, kernel_size=4, stride=4)

        self.layer_indexes = [3, 7] if clip_pretrained in ["ViT-B/16", "EVA02-CLIP-B-16"] else [7, 15]
        self.layers = []
        self.clip_name = clip_pretrained
        self.frozen_backbone(clip_finetune)

        self.enable_high_res = False

        for l in self.layer_indexes:
            if 'eva' in self.clip_name.lower():
                self.sem_seg_head.predictor.clip_model.visual.blocks[l].register_forward_hook(
                    lambda m, _, o: self.layers.append(o)
                )
            else:
                self.sem_seg_head.predictor.clip_model.visual.transformer.resblocks[l].register_forward_hook(
                    lambda m, _, o: self.layers.append(o)
                )

    @classmethod
    def from_config(cls, cfg):
        backbone = None
        sem_seg_head = build_sem_seg_head(cfg, None)
        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "clip_pixel_mean": cfg.MODEL.CLIP_PIXEL_MEAN,
            "clip_pixel_std": cfg.MODEL.CLIP_PIXEL_STD,
            "train_class_json": cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON,
            "test_class_json": cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON,
            "sliding_window": cfg.TEST.SLIDING_WINDOW,
            "clip_finetune": cfg.MODEL.SEM_SEG_HEAD.CLIP_FINETUNE,
            "backbone_multiplier": cfg.SOLVER.BACKBONE_MULTIPLIER,
            "clip_pretrained": cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED,
            "enable_high_res": cfg.MODEL.SEM_SEG_HEAD.ENABLE_HIGH_RES,
            "sw_kernel": cfg.TEST.SW_KERNEL,
            "sw_overlap": cfg.TEST.SW_OVERLAP,
            "sw_out_res": cfg.TEST.SW_OUT_RES,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    @staticmethod
    def _tokens_side_len(num_tokens: int) -> int:
        S = int(math.isqrt(num_tokens))
        if S * S != num_tokens:
            raise ValueError(f"Token count {num_tokens} is not a perfect square (got {num_tokens}).")
        return S

    def forward(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        if not self.training and self.sliding_window:
            return self.inference_sliding_window(batched_inputs)

        clip_images = [(x - self.clip_pixel_mean) / self.clip_pixel_std for x in images]
        clip_images = ImageList.from_tensors(clip_images, self.size_divisibility)

        gt_mask = torch.stack([x["sem_seg"].to(self.device) for x in batched_inputs], dim=0)

        self.layers = []
        clip_images_resized = F.interpolate(
            clip_images.tensor, size=self.clip_resolution, mode='bilinear', align_corners=False
        )

        if 'eva' in self.clip_name.lower():
            clip_features = self.sem_seg_head.predictor.clip_model.encode_dense(
                clip_images_resized, normalize=True, keep_shape=False, mode='csa',
            )  # [B, N, C]
            image_features = clip_features
        else:
            clip_features = self.sem_seg_head.predictor.clip_model.encode_image(
                clip_images_resized, dense=True
            )  # [B, 1+N, C]
            image_features = clip_features[:, 1:, :]  #  cls → [B, N, C]

        S = self._tokens_side_len(image_features.shape[1])

        res3 = rearrange(image_features, "B (H W) C -> B C H W", H=S)

        if 'eva' in self.clip_name.lower():
            res4 = rearrange(self.layers[0][:, 1:, :], "B (H W) C -> B C H W", H=S)
            res5 = rearrange(self.layers[1][:, 1:, :], "B (H W) C -> B C H W", H=S)
        else:
            res4 = rearrange(self.layers[0][1:, :, :], "(H W) B C -> B C H W", H=S)
            res5 = rearrange(self.layers[1][1:, :, :], "(H W) B C -> B C H W", H=S)

        res4 = self.upsample1(res4)
        res5 = self.upsample2(res5)

        features = {'res5': res5, 'res4': res4, 'res3': res3}
        outputs = self.sem_seg_head(clip_features, features, gt_mask=gt_mask)

        if self.training:
            targets = torch.stack([x["sem_seg"].to(self.device) for x in batched_inputs], dim=0)
            outputs = F.interpolate(
                outputs, size=(targets.shape[-2], targets.shape[-1]), mode="bilinear", align_corners=False
            )
            num_classes = outputs.shape[1]
            mask = targets != self.sem_seg_head.ignore_value
            outputs = outputs.permute(0, 2, 3, 1)  # [B, H, W, C]
            _targets = torch.zeros_like(outputs, device=self.device)
            _onehot = F.one_hot(targets[mask], num_classes=num_classes).float()
            _targets[mask] = _onehot
            loss = F.binary_cross_entropy_with_logits(outputs, _targets)
            return {"loss_sem_seg": loss}
        else:
            outputs = outputs.sigmoid()
            image_size = clip_images.image_sizes[0]
            height = batched_inputs[0].get("height", image_size[0])
            width = batched_inputs[0].get("width", image_size[1])
            output = sem_seg_postprocess(outputs[0], image_size, height, width)
            return [{'sem_seg': output}]

    @torch.no_grad()
    def inference_sliding_window(self, batched_inputs):

        def _make_starts(L: int, K: int, S: int):
            if L <= K:
                return [0]
            starts = list(range(0, L - K + 1, S))
            if starts[-1] != L - K:
                starts.append(L - K)
            return sorted(set(starts))

        torch.cuda.reset_peak_memory_stats()

        K = int(self.sw_kernel)
        overlap = float(self.sw_overlap)
        stride = int(K * (1.0 - overlap))
        assert stride > 0, "stride > 0, sw_kernel / sw_overlap"

        out_res = list(self.sw_out_res)
        img = batched_inputs[0]["image"].to(self.device, dtype=torch.float32)
        img_resized = F.interpolate(img.unsqueeze(0), size=out_res, mode="bilinear", align_corners=False).squeeze(0)
        H, W = int(img_resized.shape[-2]), int(img_resized.shape[-1])

        ys = _make_starts(H, K, stride)
        xs = _make_starts(W, K, stride)

        patches_clip, coords = [], []
        for y in ys:
            for x in xs:
                p = img_resized[:, y:y + K, x:x + K]  # [3,K,K]
                p_clip = (p - self.clip_pixel_mean) / self.clip_pixel_std
                p_clip = F.interpolate(p_clip.unsqueeze(0), size=self.clip_resolution,
                                       mode="bilinear", align_corners=False).squeeze(0)  # [3,R,R]
                patches_clip.append(p_clip)
                coords.append((y, x))

        patches_clip = torch.stack(patches_clip, dim=0).to(self.device)  # [L,3,R,R]

        self.layers = []
        if 'eva' in self.clip_name.lower():
            clip_features = self.sem_seg_head.predictor.clip_model.encode_dense(
                patches_clip, normalize=True, keep_shape=False, mode='csa',
            )  # [L, N, C]
            image_features = clip_features
        else:
            dense = self.sem_seg_head.predictor.clip_model.encode_image(
                patches_clip, dense=True
            )  # [L, 1+N, C]
            image_features = dense[:, 1:, :]  # [L, N, C]

        S = self._tokens_side_len(image_features.shape[1])
        res3 = rearrange(image_features, "B (h w) C -> B C h w", h=S)

        if 'eva' in self.clip_name.lower():
            res4 = rearrange(self.layers[0][:, 1:, :], "B (h w) C -> B C h w", h=S)
            res5 = rearrange(self.layers[1][:, 1:, :], "B (h w) C -> B C h w", h=S)
        else:
            res4 = rearrange(self.layers[0][1:, :, :], "(h w) B C -> B C h w", h=S)
            res5 = rearrange(self.layers[1][1:, :, :], "(h w) B C -> B C h w", h=S)

        res4 = self.upsample1(res4)   # ~64×64（若 S=32）
        res5 = self.upsample2(res5)   # ~128×128

        feats = {'res5': res5, 'res4': res4, 'res3': res3}
        with torch.cuda.amp.autocast():
            logits = self.sem_seg_head(image_features, feats)  # [L, C, h, w]

        probs = F.interpolate(logits, size=K, mode="bilinear", align_corners=False).sigmoid()  # [L, C, K, K]

        C = probs.size(1)
        canvas = torch.zeros(1, C, H, W, device=self.device, dtype=probs.dtype)
        weight = torch.zeros(1, 1, H, W, device=self.device, dtype=probs.dtype)
        for i, (y, x) in enumerate(coords):
            canvas[:, :, y:y + K, x:x + K] += probs[i:i + 1]
            weight[:, :, y:y + K, x:x + K] += 1
        outputs = canvas / weight.clamp_min(1e-6)  # [1, C, H, W]

        height = batched_inputs[0].get("height", out_res[0])
        width = batched_inputs[0].get("width", out_res[1])
        output = sem_seg_postprocess(outputs[0], out_res, height, width)

        return [{'sem_seg': output}]

    def frozen_backbone(self, clip_finetune):
        if 'eva' in self.clip_name.lower():
            for name, params in self.sem_seg_head.predictor.clip_model.named_parameters():
                if 'visual.blocks' in name:
                    if "attn" in name:
                        params.requires_grad = True if "q_proj" in name or "v_proj" in name else False
                    else:
                        params.requires_grad = False
                elif 'text.transformer' in name:
                    if 'resblocks' in name:
                        params.requires_grad = True if "in_proj_weight" in name else False
                    else:
                        params.requires_grad = False
                else:
                    params.requires_grad = False
        else:
            for name, params in self.sem_seg_head.predictor.clip_model.named_parameters():
                if "transformer" in name:
                    if clip_finetune == "prompt":
                        params.requires_grad = True if "prompt" in name else False
                    elif clip_finetune == "attention":
                        if "attn" in name:
                            params.requires_grad = True if "q_proj" in name or "v_proj" in name else False
                        elif "position" in name:
                            params.requires_grad = True
                        else:
                            params.requires_grad = False
                    elif clip_finetune == "full":
                        params.requires_grad = True
                    else:
                        params.requires_grad = False
                else:
                    params.requires_grad = False
