# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer
from timm.models.layers import trunc_normal_
from timm.models import create_model

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

        self.record = False


    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            x = self.fc_norm(x)
        else:
            x = self.norm(x)
            x = x[:, 0]

        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x_f = self.forward_features(x)
        self.record = x_f
        x = self.forward_head(x_f)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvBlock(nn.Module):
    expansion = 1
    def __init__(self, siz=48, end_siz=8, fin_dim=384):
        super(ConvBlock, self).__init__()
        self.planes = siz
        fin_dim = self.planes*end_siz if fin_dim != 432 else 432
        # self.bn = nn.BatchNorm2d(planes) if self.normaliz == "bn" else nn.GroupNorm(num_groups=1, num_channels=planes)
        self.stem = nn.Sequential(nn.Conv2d(3, self.planes, kernel_size=3, stride=2, padding=1),
                                  LayerNorm(self.planes, data_format="channels_first"),
                                  nn.GELU(),
                                  nn.Conv2d(self.planes, self.planes*2, kernel_size=3, stride=2, padding=1),
                                  LayerNorm(self.planes*2, data_format="channels_first"),
                                  nn.GELU(),
                                  nn.Conv2d(self.planes*2, self.planes*4, kernel_size=3, stride=2, padding=1),
                                  LayerNorm(self.planes*4, data_format="channels_first"),
                                  nn.GELU(),
                                  nn.Conv2d(self.planes*4, self.planes*8, kernel_size=3, stride=2, padding=1),
                                  LayerNorm(self.planes*8, data_format="channels_first"),
                                  nn.GELU(),
                                  nn.Conv2d(self.planes*8, fin_dim, kernel_size=1, stride=1, padding=0)
                        )
    def forward(self, x):
        out = self.stem(x)
        # out = self.bn(out)
        return out


def vit_ti(**kwargs):
    model = VisionTransformer(
        embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_small(**kwargs):
    model = VisionTransformer(
        embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_small_cvb(**kwargs):
    model = VisionTransformer(
        embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.patch_embed.proj = ConvBlock(48, end_siz=8)
    return model

def vit_small_pretrain_in1k(**kwargs):
    model = create_model('vit_small_patch16_224.augreg_in1k', pretrained=True)
    return model

def vit_small_pretrain_in21k(**kwargs):
    model = create_model('vit_small_patch16_224.augreg_in21k_ft_in1k', pretrained=True)
    return model

def vit_base_pretrain_in1k(**kwargs):
    model = create_model('vit_base_patch16_224.augreg_in1k', pretrained=True)
    return model

def vit_base_pretrain_in21k(**kwargs):
    model = create_model('vit_base_patch16_224.augreg_in21k_ft_in1k', pretrained=True)
    return model


def vit_small_timm(**kwargs):
    model = create_model('vit_small_patch16_224', pretrained=False)
    return model

def vit_base_timm(**kwargs):
    model = create_model('vit_base_patch16_224', pretrained=False)
    return model

def vit_small_ds(**kwargs):
    model = VisionTransformer(
        embed_dim=768, depth=8, num_heads=8, mlp_ratio=3, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_cvb(**kwargs):
    model = VisionTransformer(
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.patch_embed.proj = ConvBlock(48, end_siz=16, fin_dim=None)
    return model

def vit_base(**kwargs):
    model = VisionTransformer(
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_large(**kwargs):
    model = VisionTransformer(
        embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
