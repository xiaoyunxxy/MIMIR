""" Cross-Covariance Image Transformer (XCiT) in PyTorch

Paper:
    - https://arxiv.org/abs/2106.09681

Same as the official implementation, with some minor adaptations, original copyright below
    - https://github.com/facebookresearch/xcit/blob/master/xcit.py

Modifications and additions for timm hacked together by / Copyright 2021, Ross Wightman
"""
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

import math
from functools import partial

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import DropPath, trunc_normal_, to_2tuple
from timm.models._builder import build_model_with_cfg
from timm.models._features_fx import register_notrace_module
from timm.models._registry import register_model, generate_default_cfgs, register_model_deprecations
from timm.models.cait import ClassAttn
from timm.models.vision_transformer import Mlp

from util.pos_embed import get_2d_sincos_pos_embed


__all__ = ['Xcit']  # model_registry will add each entrypoint fn to this


@register_notrace_module  # reason: FX can't symbolically trace torch.arange in forward method
class PositionalEncodingFourier(nn.Module):
    """
    Positional encoding relying on a fourier kernel matching the one used in the "Attention is all you Need" paper.
    Based on the official XCiT code
        - https://github.com/facebookresearch/xcit/blob/master/xcit.py
    """

    def __init__(self, hidden_dim=32, dim=768, temperature=10000):
        super().__init__()
        self.token_projection = nn.Conv2d(hidden_dim * 2, dim, kernel_size=1)
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.dim = dim
        self.eps = 1e-6

    def forward(self, B: int, H: int, W: int):
        device = self.token_projection.weight.device
        y_embed = torch.arange(1, H+1, dtype=torch.float32, device=device).unsqueeze(1).repeat(1, 1, W)
        x_embed = torch.arange(1, W+1, dtype=torch.float32, device=device).repeat(1, H, 1)
        y_embed = y_embed / (y_embed[:, -1:, :] + self.eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + self.eps) * self.scale
        dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / self.hidden_dim)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack([pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()], dim=4).flatten(3)
        pos_y = torch.stack([pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()], dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = self.token_projection(pos)
        return pos.repeat(B, 1, 1, 1)  # (B, C, H, W)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution + batch norm"""
    return torch.nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_planes)
    )


class ConvPatchEmbed(nn.Module):
    """Image to Patch Embedding using multiple convolutional layers"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, act_layer=nn.GELU):
        super().__init__()
        img_size = to_2tuple(img_size)
        num_patches = (img_size[1] // patch_size) * (img_size[0] // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        if patch_size == 16:
            self.proj = torch.nn.Sequential(
                conv3x3(in_chans, embed_dim // 8, 2),
                act_layer(),
                conv3x3(embed_dim // 8, embed_dim // 4, 2),
                act_layer(),
                conv3x3(embed_dim // 4, embed_dim // 2, 2),
                act_layer(),
                conv3x3(embed_dim // 2, embed_dim, 2),
            )
        elif patch_size == 8:
            self.proj = torch.nn.Sequential(
                conv3x3(in_chans, embed_dim // 4, 2),
                act_layer(),
                conv3x3(embed_dim // 4, embed_dim // 2, 2),
                act_layer(),
                conv3x3(embed_dim // 2, embed_dim, 2),
            )
        else:
            raise('For convolutional projection, patch size has to be in [8, 16]')

    def forward(self, x):
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        return x, (Hp, Wp)


class LPI(nn.Module):
    """
    Local Patch Interaction module that allows explicit communication between tokens in 3x3 windows to augment the
    implicit communication performed by the block diagonal scatter attention. Implemented using 2 layers of separable
    3x3 convolutions with GeLU and BatchNorm2d
    """

    def __init__(self, in_features, out_features=None, act_layer=nn.GELU, kernel_size=3):
        super().__init__()
        out_features = out_features or in_features

        padding = kernel_size // 2

        self.conv1 = torch.nn.Conv2d(
            in_features, in_features, kernel_size=kernel_size, padding=padding, groups=in_features)
        self.act = act_layer()
        self.bn = nn.BatchNorm2d(in_features)
        self.conv2 = torch.nn.Conv2d(
            in_features, out_features, kernel_size=kernel_size, padding=padding, groups=out_features)

    def forward(self, x, H: int, W: int):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.conv1(x)
        x = self.act(x)
        x = self.bn(x)
        x = self.conv2(x)
        x = x.reshape(B, C, N).permute(0, 2, 1)
        return x


class ClassAttentionBlock(nn.Module):
    """Class Attention Layer as in CaiT https://arxiv.org/abs/2103.17239"""

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            proj_drop=0.,
            attn_drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            eta=1.,
            tokens_norm=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.attn = ClassAttn(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=proj_drop)

        if eta is not None:  # LayerScale Initialization (no layerscale when None)
            self.gamma1 = nn.Parameter(eta * torch.ones(dim))
            self.gamma2 = nn.Parameter(eta * torch.ones(dim))
        else:
            self.gamma1, self.gamma2 = 1.0, 1.0

        # See https://github.com/rwightman/pytorch-image-models/pull/747#issuecomment-877795721
        self.tokens_norm = tokens_norm

    def forward(self, x):
        x_norm1 = self.norm1(x)
        x_attn = torch.cat([self.attn(x_norm1), x_norm1[:, 1:]], dim=1)
        x = x + self.drop_path(self.gamma1 * x_attn)
        if self.tokens_norm:
            x = self.norm2(x)
        else:
            x = torch.cat([self.norm2(x[:, 0:1]), x[:, 1:]], dim=1)
        x_res = x
        cls_token = x[:, 0:1]
        cls_token = self.gamma2 * self.mlp(cls_token)
        x = torch.cat([cls_token, x[:, 1:]], dim=1)
        x = x_res + self.drop_path(x)
        return x


class XCA(nn.Module):
    """ Cross-Covariance Attention (XCA)
    Operation where the channels are updated using a weighted sum. The weights are obtained from the (softmax
    normalized) Cross-covariance matrix (Q^T \\cdot K \\in d_h \\times d_h)
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # Result of next line is (qkv, B, num (H)eads,  (C')hannels per head, N)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        
        # Paper section 3.2 l2-Normalization and temperature scaling
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # (B, H, C', N), permute -> (B, N, H, C')
        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}


class XCABlock(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            proj_drop=0.,
            attn_drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            eta=1.,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = XCA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm3 = norm_layer(dim)
        self.local_mp = LPI(in_features=dim, act_layer=act_layer)

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=proj_drop)

        self.gamma1 = nn.Parameter(eta * torch.ones(dim))
        self.gamma3 = nn.Parameter(eta * torch.ones(dim))
        self.gamma2 = nn.Parameter(eta * torch.ones(dim))

    def forward(self, x, H: int, W: int):
        x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
        # NOTE official code has 3 then 2, so keeping it the same to be consistent with loaded weights
        # See https://github.com/rwightman/pytorch-image-models/pull/747#issuecomment-877795721
        x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x


class MaskedAutoencoderXcit(nn.Module):
    """
    Based on timm and DeiT code bases
    https://github.com/rwightman/pytorch-image-models/tree/master/timm
    https://github.com/facebookresearch/deit/
    """

    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=1000,
            global_pool='token',
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.,
            pos_drop_rate=0.,
            proj_drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            act_layer=None,
            norm_layer=None,
            cls_attn_layers=2,
            use_pos_embed=True,
            eta=1.,
            tokens_norm=False,
            decoder_embed_dim=512, 
            decoder_depth=8, 
            decoder_num_heads=16,
            norm_pix_loss=False
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            drop_rate (float): dropout rate after positional embedding, and in XCA/CA projection + MLP
            pos_drop_rate: position embedding dropout rate
            proj_drop_rate (float): projection dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate (constant across all layers)
            norm_layer: (nn.Module): normalization layer
            cls_attn_layers: (int) Depth of Class attention layers
            use_pos_embed: (bool) whether to use positional encoding
            eta: (float) layerscale initialization value
            tokens_norm: (bool) Whether to normalize all tokens or just the cls_token in the CA

        Notes:
            - Although `layer_norm` is user specifiable, there are hard-coded `BatchNorm2d`s in the local patch
              interaction (class LPI) and the patch embedding (class ConvPatchEmbed)
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        img_size = to_2tuple(img_size)
        assert (img_size[0] % patch_size == 0) and (img_size[0] % patch_size == 0), \
            '`patch_size` should divide image dimensions evenly'
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.global_pool = global_pool
        self.grad_checkpointing = False
        self.num_patches = (img_size[1] // patch_size) * (img_size[0] // patch_size)

        self.patch_embed = ConvPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            act_layer=act_layer,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_pos_embed:
            self.pos_embed = PositionalEncodingFourier(dim=embed_dim)
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=pos_drop_rate)

        self.blocks = nn.ModuleList([
            XCABlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate,
                act_layer=act_layer,
                norm_layer=norm_layer,
                eta=eta,
            )
            for _ in range(depth)])

        self.cls_attn_blocks = nn.ModuleList([
            ClassAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=drop_rate,
                attn_drop=attn_drop_rate,
                act_layer=act_layer,
                norm_layer=norm_layer,
                eta=eta,
                tokens_norm=tokens_norm,
            )
            for _ in range(cls_attn_layers)])


        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_pos_embed = PositionalEncodingFourier(dim=decoder_embed_dim)

        self.decoder_blocks = nn.ModuleList([
            XCABlock(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_cls_attn_blocks = nn.ModuleList([
            ClassAttentionBlock(
                dim=decoder_embed_dim,
                num_heads=decoder_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=drop_rate,
                attn_drop=attn_drop_rate,
                act_layer=act_layer,
                norm_layer=norm_layer,
                eta=eta,
                tokens_norm=tokens_norm,
            )
            for _ in range(cls_attn_layers)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm = norm_layer(embed_dim)
        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.num_patches**.5), cls_token=True)
        # self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        # w = self.patch_embed.proj.weight.data
        # torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_encoder(self, x, mask_ratio):
        B = x.shape[0]
        # x is (B, N, C). (Hp, Hw) is (height in units of patches, width in units of patches)
        x, (Hp, Wp) = self.patch_embed(x)

        if self.pos_embed is not None:
            # `pos_embed` (B, C, Hp, Wp), reshape -> (B, C, N), permute -> (B, N, C)
            pos_encoding = self.pos_embed(B, Hp, Wp).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
            x = x + pos_encoding
        x = self.pos_drop(x)

        # if mask_ratio == 0.75 , we always use 0.75
        Hp = Hp // 2
        Wp = Wp // 2

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)


        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, Hp, Wp)
            else:
                x = blk(x, Hp, Wp)

        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)

        for blk in self.cls_attn_blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x)
            else:
                x = blk(x)

        x = self.norm(x)
        return x, mask, ids_restore

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        # x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        clst = x[:, :1, :]
        x = x_

        Hp = Wp = int(self.num_patches**.5)

        B = x.shape[0]

        # add pos embed
        # x = x + self.decoder_pos_embed
        pos_encoding = self.decoder_pos_embed(B, Hp, Wp).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
        x = x + pos_encoding

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x, Hp, Wp)
        

        x = torch.cat((clst, x), dim=1)

        for blk in self.decoder_cls_attn_blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x)
            else:
                x = blk(x)

        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75, adv_images=None):
        # readv pre-training
        # if adv_images is not None:
        #     latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        #     pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        #     loss = self.forward_loss(adv_images, pred, mask)
        #     return loss, pred, mask

        # adv pre-training
        if adv_images is not None:
            latent, mask, ids_restore = self.forward_encoder(adv_images, mask_ratio)
            pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
            loss = self.forward_loss(imgs, pred, mask)
            return loss, pred, mask, latent

        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask, latent


smaller_decoder = {
    'decoder_embed_dim': 128,
    'decoder_depth': 2,
    'decoder_num_heads': 16
}



def mae_xcit_small_dec128d2b(**kwargs):
    model = MaskedAutoencoderXcit(
        embed_dim=384, depth=12, num_heads=8, eta=1.0, tokens_norm=True,
        decoder_embed_dim=smaller_decoder['decoder_embed_dim'], 
        decoder_depth=smaller_decoder['decoder_depth'], 
        decoder_num_heads=smaller_decoder['decoder_num_heads'],
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_xcit_medium_dec128d2b(**kwargs):
    model = MaskedAutoencoderXcit(
        embed_dim=512, depth=12, num_heads=8, eta=1.0, tokens_norm=True,
        decoder_embed_dim=smaller_decoder['decoder_embed_dim'], 
        decoder_depth=smaller_decoder['decoder_depth'], 
        decoder_num_heads=smaller_decoder['decoder_num_heads'],
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_xcit_large_dec128d2b(**kwargs):
    model = MaskedAutoencoderXcit(
        embed_dim=768, depth=12, num_heads=16, eta=1.0, tokens_norm=True,
        decoder_embed_dim=smaller_decoder['decoder_embed_dim'], 
        decoder_depth=smaller_decoder['decoder_depth'], 
        decoder_num_heads=smaller_decoder['decoder_num_heads'],
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

mae_xcit_small = mae_xcit_small_dec128d2b
mae_xcit_medium = mae_xcit_medium_dec128d2b
mae_xcit_large = mae_xcit_large_dec128d2b
