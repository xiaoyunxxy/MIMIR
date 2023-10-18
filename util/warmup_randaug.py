import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
from typing import Iterable, Optional
import math


import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchattacks

import timm

from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.loader import build_dataset
from util.data import load_data, load_set


def warmup_dataloder(args, cur_epoch=0):

    # augmentation hyper parameters
    if cur_epoch == 0:
        args.aa = 'rand-m1-mstd0.5-inc1'
        args.mixup_prob=0.5
    elif cur_epoch == 1:
        args.aa = 'rand-m1-mstd0.5-inc1'
        args.mixup_prob=0.6
    elif cur_epoch == 2:
        args.aa = 'rand-m2-mstd0.5-inc1'
        args.mixup_prob=0.7
    elif cur_epoch == 3:
        args.aa = 'rand-m3-mstd0.5-inc1'
        args.mixup_prob=0.8
    elif cur_epoch == 4:
        args.aa = 'rand-m4-mstd0.5-inc1'
        args.mixup_prob=0.9
    elif cur_epoch == 5:
        args.aa = 'rand-m5-mstd0.5-inc1'
        args.mixup_prob=1.0
    elif cur_epoch == 6:
        args.aa = 'rand-m6-mstd0.5-inc1'
        args.mixup_prob=1.0
        args.mixup=0.8
        args.cutmix = 1.0
        args.mixup_switch_prob = 0.1
    elif cur_epoch == 7:
        args.aa = 'rand-m7-mstd0.5-inc1'
        args.mixup_prob=1.0
        args.cutmix = 1.0
        args.mixup_switch_prob = 0.2
    elif cur_epoch == 8:
        args.aa = 'rand-m8-mstd0.5-inc1'
        args.mixup_prob=0.9
        args.cutmix = 1.0
        args.mixup_switch_prob = 0.3
    elif cur_epoch == 9:
        args.aa = 'rand-m9-mstd0.5-inc1'
        args.mixup_prob=0.95
        args.cutmix = 1.0
        args.mixup_switch_prob = 0.4
    elif cur_epoch >= 10:
        args.aa = 'rand-m9-mstd0.5-inc1'
        args.mixup_prob=1.0
        args.cutmix = 1.0
        args.mixup_switch_prob = 0.5

    # dataset
    dataset_train = build_dataset(args, is_train=True)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)


    if args.use_edm:
        eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
        data_loader_train, _ = load_data(dataset_train, dataset_test, 
            dataset=simi_dataset, batch_size=args.batch_size, 
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

    # mixup function warm up
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)


    return data_loader_train, mixup_fn
