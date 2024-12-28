import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
from typing import Iterable, Optional
import math
import PIL

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchattacks
from torchvision import datasets, transforms
from collections import defaultdict, deque, OrderedDict

import timm
from timm.utils import accuracy

import util.misc as misc
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import util.lr_sched as lr_sched
from util.loader import ft_model_loader
from autoattack import AutoAttack

from imagenet5000_eval.loaders import CustomImageFolder


cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)


def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--patch_size', default=16, type=int,
                        help='images patch size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--warmup_aa', action='store_true')
    parser.set_defaults(warmup_aa=False)

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    parser.add_argument('--use_normalize', action='store_true',
                        help='Use normalized data for training.')
    parser.set_defaults(use_normalize=False)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--dataset', default='imagenet', type=str, help='dataset name')
    parser.add_argument('--data_root', default='../data', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--use_edm', action='store_true',
                        help='Use edm data for training.')
    parser.set_defaults(use_edm=False)
    parser.add_argument('--unsup-fraction', type=float, default=0.7, help='Ratio of unlabelled data to labelled data.')
    parser.add_argument('--aux_data_filename', type=str, help='Path to additional Tiny Images data.', 
                        default='/data/xuxx/edm_data/1m.npz')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=32, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # adversarial attack hyper parameters
    parser.add_argument('--attack_train', default='plain', type=str, help='attack type')
    parser.add_argument('--attack', default='pgd', type=str, help='attack type')
    parser.add_argument('--steps', default=10, type=int, help='adv. steps')
    parser.add_argument('--eps', default=8, type=float, help='max norm')
    parser.add_argument('--alpha', default=2, type=float, help='adv. steps size')
    parser.add_argument('--trades_beta', default=6.0, type=float, help='trades loss beta')
    parser.add_argument('--aa_file', default='eval_autoattack.txt', type=str, help='aa log file name')


    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # IB hyper parameter
    parser.add_argument('--mi_xz', default=0.0001, type=float, help='regular for mi.')
    parser.add_argument('--mi_yz', default=0.001, type=float, help='regular for mi.')
    parser.add_argument('--mi_loss', default='plain', type=str, help='Use mutual information for training.')

    # adaptive evaluation
    parser.add_argument('--adap_eval', action='store_true', help='use adaptive evaluation or not.')
    parser.set_defaults(adap_eval=False)

    return parser


def sub_imagenet():

    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    transform_test = transforms.Compose(t)
    
    data_dir = args.data_root + '/val'
    imagenet = CustomImageFolder(
        data_dir,
        transform_test)

    return imagenet


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    sub_imagenet_set = sub_imagenet()

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()

        if args.dist_eval:
            if len(sub_imagenet_set) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                sub_imagenet_set, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(sub_imagenet_set)
    else:
        sampler_val = torch.utils.data.SequentialSampler(sub_imagenet_set)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None
    
    sub_imagenet5000_eval = torch.utils.data.DataLoader(sub_imagenet_set,
        sampler=sampler_val,
        batch_size=args.batch_size,
        pin_memory=args.pin_mem,
        num_workers=args.num_workers)

    model = ft_model_loader(args)
    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    eval_load_model(args=args, model_without_ddp=model_without_ddp)

    if args.use_normalize:
        model = normalize_model(model, args.dataset)
    test_stats = evaluate(sub_imagenet5000_eval, model, device)
    print(f"Accuracy of the network on the {len(sub_imagenet5000_eval)} test images: {test_stats['acc1']:.1f}%")


    # # pgd 5 eps 1 alpha 0.5, to compare with TORA and RVT
    args.eps /= 255
    args.alpha /= 255

    # pgd 20
    print('eps: ', args.eps, '  alpha: ', args.alpha)
    evaluate_pgd(args, model, device, sub_imagenet5000_eval, eval_steps=20)

    # # pgd 100
    # print('eps: ', args.eps, '  alpha: ', args.alpha)
    # evaluate_pgd(args, model, device, eval_steps=100)

    # auto attack eval
    print('eval auto attack.')
    print('eps: ', args.eps, '  alpha: ', args.alpha)
    at_path = os.path.join(os.path.dirname(args.resume), args.aa_file)
    evaluate_aa(args, model, sub_imagenet5000_eval, at_path, device)


def normalize(data_set, X):
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2471, 0.2435, 0.2616)
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    if data_set=="cifar10":
        mu = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
        std = torch.tensor(cifar10_std).view(3, 1, 1).cuda()
    elif data_set=="imagenette" or data_set=="imagenet" or data_set=="tiny-imagenet":
        mu = torch.tensor(imagenet_mean).view(3, 1, 1).cuda()
        std = torch.tensor(imagenet_std).view(3, 1, 1).cuda()
    return (X - mu) / std

def denormalize(data_set, X):
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2471, 0.2435, 0.2616)
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    if data_set=="cifar10":
        mu = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
        std = torch.tensor(cifar10_std).view(3, 1, 1).cuda()
    elif data_set=="imagenette" or data_set=="imagenet" or data_set=="tiny-imagenet":
        mu = torch.tensor(imagenet_mean).view(3, 1, 1).cuda()
        std = torch.tensor(imagenet_std).view(3, 1, 1).cuda()
    return X*std+mu

class denormalize_model():
    def __init__(self, model, data_set):
        self.model_test = model
        self.data_set = data_set
        self.training = model.training
    def __call__(self, x):
        x_norm = normalize(self.data_set, x)
        return self.model_test(x_norm)
    def parameters(self):
        return self.model_test.parameters()
    def eval(self):
        self.model_test.eval()
    def train(self):
        self.model_test.train()

class normalize_model():
    def __init__(self, model, data_set):
        self.model_test = model
        self.data_set = data_set
        self.training = model.training
    def __call__(self, x):
        x_norm = normalize(self.data_set, x)
        return self.model_test(x_norm)
    def parameters(self):
        return self.model_test.parameters()
    def eval(self):
        self.model_test.eval()
    def train(self):
        self.model_test.train()


def evaluate_pgd(args, model, device, test_loader, eval_steps=10):
    test_loader_nonorm = test_loader

    attack = torchattacks.PGD(model, eps=args.eps,
                            alpha=args.alpha, steps=eval_steps, random_start=True)
    header='PGD {} Test:'.format(eval_steps)
    test_stats = evaluate_adv(attack, test_loader_nonorm, model, device, header)
    print(f"Accuracy of the network on the {len(test_loader_nonorm)} test images: {test_stats['acc1']:.1f}%")



def eval_load_model(args, model_without_ddp):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        try:
            model_without_ddp.load_state_dict(checkpoint)
        except:
            try:
                new_state = OrderedDict()
                for old_key in checkpoint['model_state_dict'].keys():
                    new_key = old_key.replace('module.base_model.', '')
                    new_state[new_key] = checkpoint['model_state_dict'][old_key]
                model_without_ddp.load_state_dict(new_state)
            except:
                new_state = OrderedDict()
                for old_key in checkpoint.keys():
                    new_key = old_key.replace('module.base_model.', '')
                    new_state[new_key] = checkpoint[old_key]
                model_without_ddp.load_state_dict(new_state)
                
        print("Resume checkpoint %s" % args.resume)


def evaluate_aa(args, model, test_loader, log_path, device):
    test_loader_nonorm = test_loader
    model.eval()

    # if args.use_normalize:
    #     model = normalize_model(model, args.dataset)
    #     args.eps = args.max().item()

    # evaluate with original autoattack
    l = [x for (x, y, z) in test_loader_nonorm]
    x_test = torch.cat(l, 0)
    l = [y for (x, y, z) in test_loader_nonorm]
    y_test = torch.cat(l, 0)
    
    adversary = AutoAttack(model, norm='Linf', eps=args.eps, version='standard',log_path=log_path)
    X_adv = adversary.run_standard_evaluation(x_test, y_test, bs=args.batch_size)


def evaluate_adv(attack, data_loader, model, device, header='ADV Test:'):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        adv_samples = attack(images, target)
        
        # compute output
        with torch.cuda.amp.autocast():
            output = model(adv_samples)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # adv_samples = attack(images, target)
        
        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)