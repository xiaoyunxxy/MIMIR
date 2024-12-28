import torch
import torch.nn as nn
import os
from timm.models import create_model, safe_model_name
from timm.models.layers import convert_splitbn_model, trunc_normal_

from .swin_transformer import build_swin_base, build_swin_large
from .models_vit import vit_small, vit_base, vit_large
from .pos_embed import interpolate_pos_embed

def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # Here we assume the color channel is in at dim=1

    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)

def denormalize(tensor, mean, std):
    '''
    Args:
        tensor (torch.Tensor): Float tensor image of size (B, C, H, W) to be denormalized.
        mean (torch.Tensor): float tensor means of size (C, )  for each channel.
        std (torch.Tensor): float tensor standard deviations of size (C, ) for each channel.
    '''
    return tensor*std[None]+mean[None]

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


class NormalizeByChannelMeanStd(nn.Module):
    '''The class of a normalization layer.'''
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)

class SwitchableBatchNorm2d(torch.nn.BatchNorm2d):
    '''The class of a batch norm layer.'''
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.bn_mode = 'clean'
        self.bn_adv = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input: torch.Tensor):
        if self.training:  # aux BN only relevant while training
            if self.bn_mode == 'clean':
                return super().forward(input)
            elif self.bn_mode == 'adv':
                return self.bn_adv(input)
        else:
            return super().forward(input)

def convert_switchablebn_model(module):
    """
    Recursively traverse module and its children to replace all instances of
    ``torch.nn.modules.batchnorm._BatchNorm`` with `SplitBatchnorm2d`.

    Args:
        module (torch.nn.Module): input module
        num_splits (int): number of separate batchnorm layers to split input across
    Example::
        >>> # model is an instance of torch.nn.Module
        >>> model = timm.models.convert_splitbn_model(model, num_splits=2)
    """

    mod = module
    if isinstance(module, torch.nn.modules.instancenorm._InstanceNorm):
        return module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        mod = SwitchableBatchNorm2d(
            module.num_features, module.eps, module.momentum, module.affine,
            module.track_running_stats)
        mod.running_mean = module.running_mean
        mod.running_var = module.running_var
        mod.num_batches_tracked = module.num_batches_tracked
        if module.affine:
            mod.weight.data = module.weight.data.clone().detach()
            mod.bias.data = module.bias.data.clone().detach()
            
        for aux in [mod.bn_adv]:
            aux.running_mean = module.running_mean.clone()
            aux.running_var = module.running_var.clone()
            aux.num_batches_tracked = module.num_batches_tracked.clone()
            if module.affine:
                aux.weight.data = module.weight.data.clone().detach()
                aux.bias.data = module.bias.data.clone().detach()
    for name, child in module.named_children():
        mod.add_module(name, convert_switchablebn_model(child))
    del module
    return mod

def mimir_model(args):
    if args.model.startswith('swin_base'):
        print('Loading MIMIR swin_base pretrain weights...')
        model = build_swin_base()
        mimir_ckpt = args.mimir_ckpt_path
        if not os.path.exists(mimir_ckpt):
            print('Pre-trained not exist. Return random initialized model.')
            return model
        checkpoint = torch.load(mimir_ckpt, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % mimir_ckpt)

        checkpoint_ori = checkpoint['model']
        checkpoint_model = dict()
        # remove encoder prefix
        for i in checkpoint_ori.keys():
            if i.startswith('encoder'):
                checkpoint_model[i[8:]] = checkpoint_ori[i]

        state_dict = model.state_dict()
        print('MIMIR checkpoint_model: ', checkpoint_model.keys())
        print('state_dict: ', state_dict.keys())

        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
        # trunc_normal_(model.head.weight, std=2e-5) 
    elif args.model.startswith('swin_large'):
        print('Loading MIMIR swin_large pretrain weights...')
        model = build_swin_large()
        mimir_ckpt = args.mimir_ckpt_path
        checkpoint = torch.load(mimir_ckpt, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % mimir_ckpt)

        checkpoint_ori = checkpoint['model']
        checkpoint_model = dict()
        # remove encoder prefix
        for i in checkpoint_ori.keys():
            if i.startswith('encoder'):
                checkpoint_model[i[8:]] = checkpoint_ori[i]

        state_dict = model.state_dict()
        print('MIMIR checkpoint_model: ', checkpoint_model.keys())
        print('state_dict: ', state_dict.keys())

        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
        # trunc_normal_(model.head.weight, std=2e-5) 
    elif args.model=='vit_large':
        print('Loading MIMIR vit_large pretrain weights...')
        model = vit_large(num_classes=1000,
            global_pool=True,
            img_size=224,
            patch_size=16)
        mimir_ckpt = args.mimir_ckpt_path
        checkpoint = torch.load(mimir_ckpt, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % mimir_ckpt)

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        print('MIMIR checkpoint_model: ', checkpoint_model.keys())
        print('state_dict: ', state_dict.keys())

        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
        trunc_normal_(model.head.weight, std=2e-5)
    elif args.model=='vit_base':
        print('Loading MIMIR vit_base pretrain weights...')
        model = vit_base(num_classes=1000,
            global_pool=True,
            img_size=224,
            patch_size=16)
        mimir_ckpt = args.mimir_ckpt_path
        checkpoint = torch.load(mimir_ckpt, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % mimir_ckpt)

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        print('MIMIR checkpoint_model: ', checkpoint_model.keys())
        print('state_dict: ', state_dict.keys())

        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
        trunc_normal_(model.head.weight, std=2e-5)
    elif args.model=='vit_base_cvb':
        print('Loading MIMIR vit_base_cvb pretrain weights...')
        model = vit_base(num_classes=1000,
            global_pool=True,
            img_size=224,
            patch_size=16)
        model.patch_embed.proj = ConvBlock(48, end_siz=16, fin_dim=None)

        mimir_dir = args.mimir_ckpt_path
        checkpoint = torch.load(mimir_dir, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % mimir_dir)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
        trunc_normal_(model.head.weight, std=2e-5)
        

    return model

def build_model(args, _logger, num_aug_splits):
    '''The function to build model for robust training.'''
    # creating model
    _logger.info(f"Creating model: {args.model}")
    model_kwargs=dict({
        'num_classes': args.num_classes,
        'drop_rate': args.drop,
        'drop_connect_rate': args.drop_connect,  # DEPRECATED, use drop_path
        'drop_path_rate': args.drop_path,
        'drop_block_rate': args.drop_block,
        'global_pool': args.gp,
        'bn_momentum': args.bn_momentum,
        'bn_eps': args.bn_eps,
    })
    
    
    if args.mimir:
        model = mimir_model(args)
    else:
        model = create_model(args.model, pretrained=False, **model_kwargs)

    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes
    
    _logger.info(f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')

    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    # advprop conversion
    if args.advprop:
        model=convert_switchablebn_model(model)

    model.to(args.device_id)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    # setup synchronized BatchNorm for distributed training
    if args.distributed and args.sync_bn:
        if args.amp_version == 'apex':
            # Apex SyncBN preferred unless native amp is activated
            from apex.parallel import convert_syncbn_model
            model = convert_syncbn_model(model)
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        _logger.info(
            'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
            'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

    return model

def load_pretrained_21k(args, model, logger):
    '''The function to load pretrained 21K checkpoint to 1K model.'''
    logger.info(f"==============> Loading weight {args.pretrain} for fine-tuning......")
    checkpoint = torch.load(args.pretrain, map_location='cpu')
    state_dict = checkpoint['model']

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete relative_coords_table since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic')
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = model.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C1:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                state_dict[k] = absolute_pos_embed_pretrained_resized

    # check classifier, if not match, then re-init classifier to zero
    head_bias_pretrained = state_dict['head.bias']
    Nc1 = head_bias_pretrained.shape[0]
    Nc2 = model.head.bias.shape[0]
    if (Nc1 != Nc2):
        if Nc1 == 21841 and Nc2 == 1000:
            logger.info("loading ImageNet-22K weight to ImageNet-1K ......")
            map22kto1k_path = f'data/map22kto1k.txt'
            with open(map22kto1k_path) as f:
                map22kto1k = f.readlines()
            map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
            state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
            state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
        else:
            torch.nn.init.constant_(model.head.bias, 0.)
            torch.nn.init.constant_(model.head.weight, 0.)
            del state_dict['head.weight']
            del state_dict['head.bias']
            logger.warning(f"Error in loading classifier head, re-init classifier head to 0")

    msg = model.load_state_dict(state_dict, strict=False)
    logger.warning(msg)

    logger.info(f"=> loaded successfully '{args.pretrain}'")

    del checkpoint
    torch.cuda.empty_cache()