import argparse
import datetime
import json
import numpy as np
import os
import sys
import time
import math
from pathlib import Path
from typing import Iterable
from scipy.spatial.distance import pdist, squareform

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm
import timm.optim.optim_factory as optim_factory
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import util.lr_sched as lr_sched
from util.loader import build_dataset, attack_loader
from util.data import load_data, load_set
import models_mae

# MI methods
from MI.hsic import hsic_normalized_cca
from MI.DIB_MI import calculate_MI, HSIC


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_ti_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=32, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--dataset', default='imagenet', type=str, help='dataset name')
    parser.add_argument('--data_root', default='../data', type=str,
                        help='dataset path')
    parser.add_argument('--use_edm', action='store_true',
                        help='Use edm data for training.')
    parser.set_defaults(use_edm=False)
    parser.add_argument('--unsup_fraction', type=float, default=0.7, help='Ratio of unlabelled data to labelled data.')
    parser.add_argument('--aux_data_filename', type=str, help='Path to additional Tiny Images data.', 
                        default='../data/edm/1m.npz')

    parser.add_argument('--output_dir', default='./experiment',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./experiment',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=32, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # adversarial attack hyper parameter
    parser.add_argument('--attack', default='plain', type=str, help='attack type')
    parser.add_argument('--steps', default=10, type=int, help='adv. steps')
    parser.add_argument('--eps', default=8/255, type=float, help='max norm')
    parser.add_argument('--alpha', default=2/255, type=float, help='adv. steps size')

    # IB hyper parameter
    parser.add_argument('--mi_xl', default=0.0, type=float, help='regular for mi.')
    parser.add_argument('--mi_xpl', default=0.00001, type=float, help='regular for mi.')
    parser.add_argument('--mi_train', default='plain', type=str, help='Use mutual information for training.')


    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


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

    if args.dataset=='imagenet':
        # simple augmentation
        transform_train = transforms.Compose([
                transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)])
        dataset_train = datasets.ImageFolder(os.path.join(args.data_root, 'train'), transform=transform_train)
    elif args.use_edm:
        args.dataset = args.dataset + 's'
        dataset_train, dataset_test = load_set(args.dataset, args.data_root, batch_size=args.batch_size, batch_size_test=128, 
        num_workers=args.num_workers, aux_data_filename=args.aux_data_filename, unsup_fraction=args.unsup_fraction)
    else:
        dataset_train = build_dataset(args, is_train=True)

    if True: # args.distributed
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    if args.use_edm:
        eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
        data_loader_train, _ = load_data(dataset_train, dataset_test, 
            dataset=args.dataset, batch_size=args.batch_size, 
            batch_size_test=128, eff_batch_size=eff_batch_size, 
            num_workers=args.num_workers, 
            aux_data_filename=args.aux_data_filename, 
            unsup_fraction=args.unsup_fraction,
            num_replicas=num_tasks, rank=global_rank)
        del dataset_test
    else:
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
            persistent_workers=True
        )
    
    # define the model
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

    model.to(device)

    model_without_ddp = model
    # print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # define adversarial attack
    if args.attack!='plain':
        if args.dataset == 'imagenet':
            mu = torch.tensor(IMAGENET_DEFAULT_MEAN).view(3, 1, 1).to(device)
            std = torch.tensor(IMAGENET_DEFAULT_STD).view(3, 1, 1).to(device)
            upper_limit = ((1 - mu) / std)
            lower_limit = ((0 - mu) / std)
        else:
            upper_limit = 1
            lower_limit = 0
        args.eps *= upper_limit - lower_limit
        args.alpha *= upper_limit - lower_limit
        attack = attack_loader(args, model, upper_limit=upper_limit, lower_limit=lower_limit)
    else:
        attack = None

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=False)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            if args.use_edm:
                data_loader_train.batch_sampler.set_epoch(epoch)
            else:
                data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            attack=attack,
            args=args
        )
        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None, attack=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            if attack is not None:
                adv_images = attack(samples, targets)
                loss, pred, _, latent = model(samples, mask_ratio=args.mask_ratio, adv_images=adv_images)
                
                if args.mi_train=='hsic':
                    latent = latent.view(latent.shape[0], -1)

                    h_data_adv = adv_images.view(adv_images.shape[0], -1)
                    h_data = samples.view(samples.shape[0], -1)

                    # h_x_l = hsic_normalized_cca(latent, h_data, sigma=5)
                    h_xp_l = hsic_normalized_cca(latent, h_data_adv, sigma=5)

                    hsic_loss = args.mi_xpl * h_xp_l
                    if math.isfinite(hsic_loss):
                        loss += hsic_loss
                    else:
                        print("hsic is {}, skipping hisc loss".format(hsic_loss))
                elif args.mi_train=='dib_mi':
                    with torch.no_grad():
                        Z = latent.view(latent.shape[0], -1)
                        sigma_z = torch.sort(torch.cdist(Z,Z,p=2))[0][0:10].mean()

                        inputs = samples.view(samples.shape[0], -1)
                        sigma_input = torch.sort(torch.cdist(inputs, inputs,p=2))[0][0:10].mean()

                    I_Xp_Z = calculate_MI(inputs, Z, s_x=sigma_input, s_y=sigma_z)
                    dib_loss = args.mi_xpl * I_Xp_Z
                    loss += dib_loss
                else:
                    pass
            else:
                loss, _, _, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            if args.mi_train=='hsic':
                log_writer.add_scalar('mi', h_xp_l.item(), epoch_1000x)
            elif args.mi_train=='dib_mi':
                log_writer.add_scalar('mi', I_Xp_Z.item(), epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
