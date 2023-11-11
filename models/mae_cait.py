from functools import partial

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, use_fused_attn
from timm.models._manipulate import checkpoint_seq
from timm.models.cait import LayerScaleBlock, LayerScaleBlockClassAttn, TalkingHeadAttn, ClassAttn
from timm.models.vision_transformer import Block

from util.pos_embed import get_2d_sincos_pos_embed


class MaskedAutoencoderCait(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to adapt to our cait models
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
            block_layers=LayerScaleBlock,
            block_layers_token=LayerScaleBlockClassAttn,
            patch_layer=PatchEmbed,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            attn_block=TalkingHeadAttn,
            mlp_block=Mlp,
            init_values=1e-4,
            attn_block_token_only=ClassAttn,
            mlp_block_token_only=Mlp,
            depth_token_only=2,
            mlp_ratio_token_only=4.0,
            decoder_embed_dim=512, 
            decoder_depth=8, 
            decoder_num_heads=16,
            norm_pix_loss=False,
            use_cait_block=True
    ):
        super().__init__()
        assert global_pool in ('', 'token', 'avg')

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim
        self.grad_checkpointing = False
        self.use_cait_block = use_cait_block

        self.patch_embed = patch_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=pos_drop_rate)

        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.Sequential(*[block_layers(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_drop=proj_drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[i],
            norm_layer=norm_layer,
            act_layer=act_layer,
            attn_block=attn_block,
            mlp_block=mlp_block,
            init_values=init_values,
        ) for i in range(depth)])

        self.blocks_token_only = nn.ModuleList([block_layers_token(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio_token_only,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            act_layer=act_layer,
            attn_block=attn_block_token_only,
            mlp_block=mlp_block_token_only,
            init_values=init_values,
        ) for _ in range(depth_token_only)])

        self.norm = norm_layer(embed_dim)

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        if self.use_cait_block:
            de_dpr = [drop_path_rate for i in range(decoder_depth)]
            self.decoder_blocks = nn.Sequential(*[block_layers(
                dim=decoder_embed_dim,
                num_heads=decoder_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=de_dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                attn_block=attn_block,
                mlp_block=mlp_block,
                init_values=init_values,
            ) for i in range(decoder_depth)])
        else:
            self.decoder_blocks = nn.ModuleList([
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x, mask_ratio):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        for i, blk in enumerate(self.blocks_token_only):
            cls_tokens = blk(x, cls_tokens)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        
        # add pos embed
        x_ = x_ + self.decoder_pos_embed

        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # apply Transformer blocks
        # for blk in self.decoder_blocks:
        #     x = blk(x)
        x = self.decoder_blocks(x)
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

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
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
        p = self.patch_embed.patch_size[0]
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


smaller_decoder = {
    'decoder_embed_dim': 128,
    'decoder_depth': 2,
    'decoder_num_heads': 16
}


model_args_xxs24_moreheads = dict(patch_size=16, embed_dim=192, depth=24, num_heads=12, init_values=1e-5)

model_args_xxs24 = dict(patch_size=16, embed_dim=192, depth=24, num_heads=4, init_values=1e-5)
model_args_xxs36 = dict(patch_size=16, embed_dim=192, depth=36, num_heads=4, init_values=1e-5)
model_args_s36 = dict(patch_size=16, embed_dim=384, depth=36, num_heads=8, init_values=1e-6)

def mae_cait_xxs24_mh_dec128d2b(**kwargs):
    model = MaskedAutoencoderCait(
        embed_dim=model_args_xxs24_moreheads['embed_dim'], 
        depth=model_args_xxs24_moreheads['depth'], 
        num_heads=model_args_xxs24_moreheads['num_heads'],
        init_values=model_args_xxs24_moreheads['init_values'],
        decoder_embed_dim=smaller_decoder['decoder_embed_dim'], 
        decoder_depth=smaller_decoder['decoder_depth'], 
        decoder_num_heads=smaller_decoder['decoder_num_heads'],
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_cait_xxs24_dec128d2b(**kwargs):
    model = MaskedAutoencoderCait(
        embed_dim=model_args_xxs24['embed_dim'], 
        depth=model_args_xxs24['depth'], 
        num_heads=model_args_xxs24['num_heads'],
        init_values=model_args_xxs24['init_values'],
        decoder_embed_dim=smaller_decoder['decoder_embed_dim'], 
        decoder_depth=smaller_decoder['decoder_depth'], 
        decoder_num_heads=smaller_decoder['decoder_num_heads'],
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_cait_xxs36_dec128d2b(**kwargs):
    model = MaskedAutoencoderCait(
        embed_dim=model_args_xxs36['embed_dim'], 
        depth=model_args_xxs36['depth'], 
        num_heads=model_args_xxs36['num_heads'],
        init_values=model_args_xxs36['init_values'],
        decoder_embed_dim=smaller_decoder['decoder_embed_dim'], 
        decoder_depth=smaller_decoder['decoder_depth'], 
        decoder_num_heads=smaller_decoder['decoder_num_heads'],
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_cait_s36_dec128d2b(**kwargs):
    model = MaskedAutoencoderCait(
        embed_dim=model_args_s36['embed_dim'], 
        depth=model_args_s36['depth'], 
        num_heads=model_args_s36['num_heads'],
        init_values=model_args_s36['init_values'],
        decoder_embed_dim=smaller_decoder['decoder_embed_dim'], 
        decoder_depth=smaller_decoder['decoder_depth'], 
        decoder_num_heads=smaller_decoder['decoder_num_heads'],
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

mae_cait_xxs24_mh=mae_cait_xxs24_mh_dec128d2b
mae_cait_xxs24=mae_cait_xxs24_dec128d2b
mae_cait_xxs36=mae_cait_xxs36_dec128d2b
mae_cait_s36=mae_cait_s36_dec128d2b

