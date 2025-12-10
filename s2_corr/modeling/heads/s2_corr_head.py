# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import math
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union

from einops import rearrange
import fvcore.nn.weight_init as weight_init
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY
from ..mamba.s2_corr_predictor import S2_CorrPredictor



@SEM_SEG_HEADS_REGISTRY.register()
class S2_CorrHead(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        num_classes: int,
        ignore_value: int = -1,
        # extra parameters
        feature_resolution: list,
        transformer_predictor: nn.Module,
        clip_pretrained: str,
    ):
        super().__init__()
        self.ignore_value = ignore_value
        self.predictor = transformer_predictor
        self.num_classes = num_classes
        self.feature_resolution = feature_resolution
        self.clip_name = clip_pretrained

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        return {
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "feature_resolution": cfg.MODEL.SEM_SEG_HEAD.FEATURE_RESOLUTION,
            "transformer_predictor": S2_CorrPredictor(cfg),
            "clip_pretrained": cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED,
        }

    @staticmethod
    def _infer_side_len(n_tokens: int) -> int:
        s = int(math.isqrt(n_tokens))
        if s * s != n_tokens:
            raise ValueError(f"[S2_CorrHead] token count {n_tokens} is not a perfect square.")
        return s

    def forward(self, features, guidance_features, prompt=None, gt_cls=None, gt_mask=None):

        if 'eva' in self.clip_name.lower():
            img_tokens = features  # [B, N, C]
        else:
            if features.dim() != 3 or features.size(1) < 2:
                raise RuntimeError(f"Unexpected features shape: {features.shape}")
            img_tokens = features[:, 1:, :]  # [B, N, C]

        B, N, C = img_tokens.shape
        S = self._infer_side_len(N)
        img_feat = rearrange(img_tokens, "b (h w) c -> b c h w", h=S, w=S)

        return self.predictor(img_feat, guidance_features, prompt, gt_cls, gt_mask)
