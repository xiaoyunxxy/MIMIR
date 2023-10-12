#!/usr/bin/env python
import os
import PIL

# torch package
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

import torchattacks
from pgd_mae import pgd_mae, pgd
from autoattack import AutoAttack


def attack_loader(args, net, upper_limit=1, lower_limit=0):
    # Gradient Clamping based Attack
    if args.attack == "pgd":
        return pgd.PGD(model=net, eps=args.eps,
                                alpha=args.alpha, steps=args.steps, random_start=True,
                                upper_limit=upper_limit, lower_limit=lower_limit)

    elif args.attack == "cw":
        return torchattacks.CW(model=net, steps=args.steps)

    elif args.attack == "auto":
        adversary = AutoAttack(model, norm='Linf', eps=args.eps, 
            log_path=args.log_dir, version='standard', seed=args.seed)
        return adversary

    elif args.attack == "pgd_mae":
        return pgd_mae.PGD_MAE(model=net, eps=args.eps,
                                alpha=args.alpha, steps=args.steps, random_start=True,
                                upper_limit=upper_limit, lower_limit=lower_limit)

# load 3 attacks for validation
def attacks_loader(args, net, device):
    attack_pgd = torchattacks.PGD(net, eps=args.eps,
                                alpha=args.alpha, steps=args.steps, random_start=True)

    attack_cw = torchattacks.CW(net, steps=args.steps)

    aa = torchattacks.AutoAttack(net, norm='Linf', eps=args.eps, 
        version='standard', seed=args.seed)

    return attack_pgd, attack_cw, aa



def dataset_transforms(args, is_train):

    # Setting Dataset Required Parameters
    if args.dataset   == "svhn":
        args.nb_classes = 10
        args.input_size= 32
        args.channel   = 3
    elif args.dataset == "cifar10":
        args.nb_classes = 10
        args.input_size= 32
        args.channel   = 3
    elif args.dataset == "tiny":
        args.nb_classes = 200
        args.input_size= 64
        args.channel   = 3
    elif args.dataset == "cifar100":
        args.nb_classes = 100
        args.input_size= 32
        args.channel   = 3
    elif args.dataset == "imagenet":
        args.nb_classes = 1000
        args.input_size= 224
        args.channel   = 3
    elif args.dataset == "imagenette":
        args.nb_classes= 10
        args.input_size= 224
        args.channel   = 3

    if args.dataset == "imagenet":
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD

        # train transform
        transform_train = create_transform(
                input_size=args.input_size,
                is_training=True,
                color_jitter=args.color_jitter,
                auto_augment=args.aa,
                interpolation='bicubic',
                re_prob=args.reprob,
                re_mode=args.remode,
                re_count=args.recount,
                mean=mean,
                std=std,
            )

        # eval transform
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
        # t.append(transforms.Normalize(mean, std))
        transform_test = transforms.Compose(t)
    else:
        transform_train = transforms.Compose(
            [transforms.RandomCrop(args.input_size, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor()]
        )
        transform_test = transforms.Compose(
            [transforms.ToTensor()]
        )

    if is_train:
        return transform_train
    else:
        return transform_test


def build_dataset(args, is_train):
        transform = dataset_transforms(args, is_train=is_train)

        if args.dataset == "cifar10":
            return torchvision.datasets.CIFAR10(root=args.data_root, transform=transform, download=True, train=is_train)
        elif args.dataset == "cifar100":
            return torchvision.datasets.CIFAR100(root=args.data_root, transform=transform, download=True, train=is_train)
        elif args.dataset == "svhn":
            return torchvision.datasets.SVHN(root=args.data_root, transform=transform, download=True,
                                    split="train" if train else "test")
        elif args.dataset == "tiny":
            return torchvision.datasets.ImageFolder(root=args.data_root+'/tiny-imagenet-200/train' if is_train \
                                    else args.data_root + '/tiny-imagenet-200/valid', transform=transform)
        elif args.dataset == "imagenet":
            return torchvision.datasets.ImageFolder(root=args.data_root+'/imagenet/train' if is_train \
                                    else args.data_root + '/imagenet/val', transform=transform)
        elif args.dataset == "imagenette":
            return torchvision.datasets.ImageFolder(root=args.data_root+'/imagenette2/train' if is_train \
                                    else args.data_root + '/imagenette2/val', transform=transform)