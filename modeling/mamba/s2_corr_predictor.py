# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Seokju Cho from: https://github.com/cvlab-kaist/CAT-Seg

import fvcore.nn.weight_init as weight_init
import open_clip.eva_clip
import torch

from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d

from .model import S2_Corr

from s2_corr.third_party import clip
from s2_corr.third_party import imagenet_templates

import numpy as np
import open_clip
import open_clip.eva_clip as eva_clip
import json


class S2_CorrPredictor(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        train_class_json: str,
        test_class_json: str,
        clip_pretrained: str,
        cache_dir: str,
        prompt_ensemble_type: str,
        text_guidance_dim: int,
        text_guidance_proj_dim: int,
        appearance_guidance_dim: int,
        appearance_guidance_proj_dim: int,
        prompt_depth: int,
        prompt_length: int,
        decoder_dims: list,
        decoder_guidance_dims: list,
        decoder_guidance_proj_dims: list,
        num_heads: int,
        num_layers: tuple,
        hidden_dims: tuple,
        pooling_sizes: tuple,
        feature_resolution: tuple,
        chunk_sizes: tuple,
        gamma: tuple
    ):

        super().__init__()
        

        # use class_texts in train_forward, and test_class_texts in test_forward
        with open(train_class_json, 'r') as f_in:
            self.class_texts = json.load(f_in)
        with open(test_class_json, 'r') as f_in:
            self.test_class_texts = json.load(f_in)
        assert self.class_texts != None
        if self.test_class_texts == None:
            self.test_class_texts = self.class_texts

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        if clip_pretrained == "ViT-G" or clip_pretrained == "ViT-H":
            # for OpenCLIP models
            name, pretrain = ('ViT-H-14', 'laion2b_s32b_b79k') if clip_pretrained == 'ViT-H' else ('ViT-bigG-14', 'laion2b_s39b_b160k')
            clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
                name, 
                pretrained=pretrain, 
                device=device, 
                force_image_size=336,)
        
            self.tokenizer = open_clip.get_tokenizer(name)
        elif clip_pretrained=="EVA02-CLIP-B-16" or clip_pretrained=="EVA02-CLIP-L-14-336":
            clip_model = eva_clip.create_model(model_name=clip_pretrained,
                                                pretrained=cache_dir, 
                                                force_custom_clip=True,
                                                precision="amp",
                                                device=device)
            self.tokenizer = open_clip.get_tokenizer(clip_pretrained)
            clip_preprocess=None
        else:
            # for OpenAI models
            clip_model, clip_preprocess = clip.load(clip_pretrained, device=device, jit=False, prompt_depth=prompt_depth, prompt_length=prompt_length)
        self.prompt_ensemble_type = prompt_ensemble_type   
        # ---- Prompt template selection ----
        if self.prompt_ensemble_type == "imagenet_select":
            prompt_templates = imagenet_templates.IMAGENET_TEMPLATES_SELECT
        elif self.prompt_ensemble_type == "imagenet":
            prompt_templates = imagenet_templates.IMAGENET_TEMPLATES
        elif self.prompt_ensemble_type == "single":
            prompt_templates = ['A photo of a {} in the scene']
        elif self.prompt_ensemble_type == "multi_domain":
            prompt_templates = [
                "A photo of {} in different environments",
                "An image of {} under various conditions",
                "{} in an outdoor scene",
                "A view of {} with changing lighting",
                "A picture of {} in rainy or foggy weather",
                "{} in a night scene",
                "{} on a city street or highway",
                "A scene that contains {}",
                "A realistic image showing {}",
                "{} appearing in an unseen domain",
            ]
        else:
            raise NotImplementedError(f"Unknown prompt_ensemble_type: {self.prompt_ensemble_type}")

        
        self.prompt_templates = prompt_templates
        self.text_features = self.class_embeddings(self.class_texts, prompt_templates, clip_model).permute(1, 0, 2).float()
        self.text_features_test = self.class_embeddings(self.test_class_texts, prompt_templates, clip_model).permute(1, 0, 2).float()
        
        self.clip_model = clip_model.float()
        self.clip_preprocess = clip_preprocess
        Corr = S2_Corr(
            text_guidance_dim=text_guidance_dim,
            text_guidance_proj_dim=text_guidance_proj_dim,
            appearance_guidance_dim=appearance_guidance_dim,
            appearance_guidance_proj_dim=appearance_guidance_proj_dim,
            decoder_dims=decoder_dims,
            decoder_guidance_dims=decoder_guidance_dims,
            decoder_guidance_proj_dims=decoder_guidance_proj_dims,
            num_layers=num_layers,
            nheads=num_heads,
            hidden_dim=hidden_dims,
            pooling_size=pooling_sizes,
            feature_resolution=feature_resolution,
            chunk_size=chunk_sizes,
            pad_len=0,
            gamma_range=gamma
        )

        self.Corr = Corr
        self.tokens = None
        self.cache = None


    @classmethod
    def from_config(cls, cfg):#, in_channels, mask_classification):
        ret = {}

        ret["train_class_json"] = cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON
        ret["test_class_json"] = cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON
        ret["clip_pretrained"] = cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED
        ret["cache_dir"] = cfg.MODEL.SEM_SEG_HEAD.CACHE_DIR
        ret["prompt_ensemble_type"] = cfg.MODEL.PROMPT_ENSEMBLE_TYPE

        # Aggregator parameters:
        ret["text_guidance_dim"] = cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_DIM
        ret["text_guidance_proj_dim"] = cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_PROJ_DIM
        ret["appearance_guidance_dim"] = cfg.MODEL.SEM_SEG_HEAD.APPEARANCE_GUIDANCE_DIM
        ret["appearance_guidance_proj_dim"] = cfg.MODEL.SEM_SEG_HEAD.APPEARANCE_GUIDANCE_PROJ_DIM

        ret["decoder_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_DIMS
        ret["decoder_guidance_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_DIMS
        ret["decoder_guidance_proj_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_PROJ_DIMS

        ret["prompt_depth"] = cfg.MODEL.SEM_SEG_HEAD.PROMPT_DEPTH
        ret["prompt_length"] = cfg.MODEL.SEM_SEG_HEAD.PROMPT_LENGTH

        ret["num_layers"] = cfg.MODEL.SEM_SEG_HEAD.NUM_LAYERS
        ret["num_heads"] = cfg.MODEL.SEM_SEG_HEAD.NUM_HEADS
        ret["hidden_dims"] = cfg.MODEL.SEM_SEG_HEAD.HIDDEN_DIMS
        ret["pooling_sizes"] = cfg.MODEL.SEM_SEG_HEAD.POOLING_SIZES
        ret["feature_resolution"] = cfg.MODEL.SEM_SEG_HEAD.FEATURE_RESOLUTION
        ret["chunk_sizes"] = cfg.MODEL.SEM_SEG_HEAD.CHUNK_SIZES
        ret["gamma"] = cfg.MODEL.SEM_SEG_HEAD.GAMMA
        return ret

    def forward(self, x, vis_guidance, prompt=None, gt_cls=None, gt_mask=None):
        vis = [vis_guidance[k] for k in vis_guidance.keys()][::-1]
        text = self.class_texts if self.training else self.test_class_texts
        text = [text[c] for c in gt_cls] if gt_cls is not None else text
        text = self.get_text_embeds(text, self.prompt_templates, self.clip_model, prompt)
        text = text.repeat(x.shape[0], 1, 1, 1)

        out = self.Corr(x, text, vis)
        return out

    @torch.no_grad()
    def class_embeddings(self, classnames, templates, clip_model):
        zeroshot_weights = []
        for classname in classnames:
            if ', ' in classname:
                classname_splits = classname.split(', ')
                texts = []
                for template in templates:
                    for cls_split in classname_splits:
                        texts.append(template.format(cls_split))
            else:
                texts = [template.format(classname) for template in templates]  # format with class
            if self.tokenizer is not None:
                texts = self.tokenizer(texts).cuda()
            else: 
                texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            if len(templates) != class_embeddings.shape[0]:
                class_embeddings = class_embeddings.reshape(len(templates), -1, class_embeddings.shape[-1]).mean(dim=1)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights

    def get_text_embeds(self, classnames, templates, clip_model, prompt=None):
        #### Fix bugs
        if self.cache is not None and not self.training:
            return self.cache

        tokens_list = []
        for classname in classnames:
            if ', ' in classname:
                classname_splits = classname.split(', ')
                texts = [template.format(classname_splits[0]) for template in templates]
            else:
                texts = [template.format(classname) for template in templates]

            if self.tokenizer is not None:
                tok = self.tokenizer(texts).cuda()
            else:
                tok = clip.tokenize(texts).cuda()
            tokens_list.append(tok)  # [M, ctx_len]

        #  [N, M, ctx_len]，reshape [N*M, ctx_len]
        tokens = torch.stack(tokens_list, dim=0)  # [N, M, ctx_len]
        N, M, L = tokens.shape
        tokens_flat = tokens.view(N * M, L)

        # 3)  [N*M, D]  → reshape [N, M, D]
        class_embeddings = clip_model.encode_text(tokens_flat, prompt)  # [N*M, D]
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        D = class_embeddings.shape[-1]
        class_embeddings = class_embeddings.view(N, M, D).contiguous()  # [N, M, D]

        # 4)  [N, M, D]
        if not self.training:
            self.cache = class_embeddings

        return class_embeddings


