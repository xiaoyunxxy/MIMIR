from functools import partial

import torch
import torch.nn as nn

from timm.layers import DropPath, trunc_normal_, PatchEmbed, Mlp, LayerNorm
from timm.models.vision_transformer_hybrid import HybridEmbed
from timm.models.convit import Block
import timm.models.convit


def convit_ti(**kwargs):
    model = timm.models.convit.convit_tiny(**kwargs)
    return model

def convit_small(**kwargs):
    model = timm.models.convit.convit_small(**kwargs)
    return model

def convit_base(**kwargs):
    model = model = timm.models.convit.convit_base(**kwargs)
    return model