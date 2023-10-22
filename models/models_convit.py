from functools import partial

import torch
import torch.nn as nn

from timm.layers import DropPath, trunc_normal_, PatchEmbed, Mlp, LayerNorm
from timm.models.vision_transformer_hybrid import HybridEmbed
from timm.models.convit import Block
import timm.models.convit

class ConVit(timm.models.convit.ConVit):
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

model_args_ti = dict(
        local_up_to_layer=10, locality_strength=1.0, embed_dim=48, num_heads=4)

model_args_small = dict(
        local_up_to_layer=10, locality_strength=1.0, embed_dim=48, num_heads=9)

model_args_base = dict(
        local_up_to_layer=10, locality_strength=1.0, embed_dim=48, num_heads=16)

def convit_ti(**kwargs):
    model = ConVit(local_up_to_layer=model_args_small['local_up_to_layer'],
    	locality_strength=model_args_small['locality_strength'],
        embed_dim=model_args_small['embed_dim'], 
        num_heads=model_args_small['num_heads'],
        qkv_bias=True, **kwargs)
    return model