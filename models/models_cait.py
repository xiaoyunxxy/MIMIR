from timm.models.cait import Cait
import timm.models.cait

model_args_xxs24 = dict(patch_size=16, embed_dim=192, depth=24, num_heads=4, init_values=1e-5)
model_args_xxs36 = dict(patch_size=16, embed_dim=192, depth=36, num_heads=4, init_values=1e-5)

def cait_xxs24(**kwargs):
    model = timm.models.cait.cait_xxs24_224(**kwargs)
    return model

def cait_xxs36(**kwargs):
    model = timm.models.cait.cait_xxs36_224(**kwargs)
    return model