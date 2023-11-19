import torch 
from timm.models.xcit import register_model, _create_xcit, Xcit


def adapt_model_patches(model: Xcit, new_patch_size: int):
    to_divide = model.patch_embed.patch_size / new_patch_size
    assert int(to_divide) == to_divide, "The new patch size should divide the original patch size"
    to_divide = int(to_divide)
    assert to_divide % 2 == 0, "The ratio between the original patch size and the new patch size should be divisible by 2"
    for conv_index in range(0, to_divide, 2):
        model.patch_embed.proj[conv_index][0].stride = (1, 1)
    model.patch_embed.patch_size = new_patch_size
    return model


@register_model
def xcit_small_12_p16_224(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=16, embed_dim=384, depth=12, num_heads=8, eta=1.0, tokens_norm=True)
    model = _create_xcit('xcit_small_12_p16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def xcit_medium_12_p16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16,
                        embed_dim=512,
                        depth=12,
                        num_heads=8,
                        eta=1.0,
                        tokens_norm=True,
                        **kwargs)
    model = _create_xcit('xcit_medium_12_p16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def xcit_large_12_p16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16,
                        embed_dim=768,
                        depth=12,
                        num_heads=16,
                        eta=1.0,
                        tokens_norm=True,
                        **kwargs)
    model = _create_xcit('xcit_large_12_p16_224', pretrained=pretrained, **model_kwargs)
    return model


xcit_small = xcit_small_12_p16_224
xcit_medium = xcit_medium_12_p16_224
xcit_large = xcit_large_12_p16_224