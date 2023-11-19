from timm.models.cait import Cait
import timm.models.cait
from functools import partial
import torch.nn as nn
import torch

model_args_xxs24_mh = dict(patch_size=16, embed_dim=192, depth=24, num_heads=12, init_values=1e-5)

model_args_xxs24 = dict(patch_size=16, embed_dim=192, depth=24, num_heads=4, init_values=1e-5)
model_args_xxs36 = dict(patch_size=16, embed_dim=192, depth=36, num_heads=4, init_values=1e-5)

def cait_xxs24_mh(**kwargs):
    model = Cait(
        embed_dim=model_args_xxs24_mh['embed_dim'], 
        depth=model_args_xxs24_mh['depth'], 
        num_heads=model_args_xxs24_mh['num_heads'], 
        init_values=model_args_xxs24_mh['init_values'],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def cait_xxs24(**kwargs):
    model = timm.models.cait.cait_xxs24_224(**kwargs)
    return model

def cait_xxs36(**kwargs):
    model = timm.models.cait.cait_xxs36_224(**kwargs)
    return model

def cait_s36(**kwargs):
    model = timm.models.cait.cait_s36_384(**kwargs)
    return model