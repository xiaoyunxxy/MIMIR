from timm.models.cait import Cait
import timm.models.cait

model_args_xxs24 = dict(patch_size=16, embed_dim=192, depth=24, num_heads=4, init_values=1e-5)
model_args_xxs36 = dict(patch_size=16, embed_dim=192, depth=36, num_heads=4, init_values=1e-5)

def cait_xxs24(**kwargs):
    model = Cait(
        embed_dim=model_args_xxs24['embed_dim'], 
        depth=model_args_xxs24['depth'], 
        num_heads=model_args_xxs24['num_heads'],
        init_values=model_args_xxs24['init_values'],
        mlp_ratio=4, **kwargs)
    return model

def cait_xxs36(**kwargs):
    model = Cait(
        embed_dim=model_args_xxs36['embed_dim'], 
        depth=model_args_xxs36['depth'], 
        num_heads=model_args_xxs36['num_heads'],
        init_values=model_args_xxs36['init_values'],
        mlp_ratio=4, **kwargs)
    return model