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
from util.data import load_data, load_set

import models.mae_vit as mae_vit
import models.mae_convit as mae_convit
import models.mae_cait as mae_cait

import models.models_vit as models_vit
import models.models_convit as models_convit
import models.models_cait as models_cait

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

def pre_model_loader(args):
    if args.model.startswith('mae_vit'):
        model = mae_vit.__dict__[args.model](
            norm_pix_loss=args.norm_pix_loss,
            img_size=args.input_size,
            patch_size=args.patch_size)
    elif args.model.startswith('mae_convit'):
        model = mae_convit.__dict__[args.model](
            norm_pix_loss=args.norm_pix_loss,
            img_size=args.input_size,
            patch_size=args.patch_size)
    elif args.model.startswith('mae_cait'):
        model = mae_cait.__dict__[args.model](
            norm_pix_loss=args.norm_pix_loss,
            img_size=args.input_size,
            patch_size=args.patch_size)

    return model

def ft_model_loader(args):
    if args.model.startswith('vit'):
        model = models_vit.__dict__[args.model](
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
            img_size=args.input_size,
            patch_size=args.patch_size)
    elif args.model.startswith('convit'):
        model = models_convit.__dict__[args.model](
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            img_size=args.input_size,
            patch_size=args.patch_size)
    elif args.model.startswith('cait'):
        model = models_cait.__dict__[args.model](
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            img_size=args.input_size,
            patch_size=args.patch_size)

    return model


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

    if args.dataset=="imagenet" or args.dataset=="imagenette":
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD
    elif args.dataset=="cifar10" or args.dataset=="cifar10s":
        mean = cifar10_mean
        std = cifar10_std
    else:
        print('checking dataset for mean and std!')
        exit(0)

    if args.aa!='noaug' and args.use_normalize:
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
            std=std)
    else:
        if args.use_normalize:
            if args.dataset=='imagenet' or args.dataset=='imagenette':
                transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)])
            else: 
                transform_train = transforms.Compose(
                    [transforms.RandomCrop(args.input_size, padding=4),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     transforms.Normalize(mean, std)]
                )
        else:
            # not using normalization 
            if args.dataset=='imagenet':
                transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()])
            else:
                transform_train = transforms.Compose(
                    [transforms.RandomCrop(args.input_size, padding=4),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor()])
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
    t.append(transforms.Normalize(mean, std))
    if args.use_normalize:
        # eval transform
        transform_test = transforms.Compose(t)
    else:
        transform_test = transforms.Compose(t[0:-1])

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
            return torchvision.datasets.ImageFolder(root=args.data_root+'/train' if is_train \
                                    else args.data_root + '/val', transform=transform)
        elif args.dataset == "imagenette":
            return torchvision.datasets.ImageFolder(root=args.data_root+'/imagenette2/train' if is_train \
                                    else args.data_root + '/imagenette2/val', transform=transform)
        elif args.dataset == "cifar10s":
            dataset_train, dataset_test = load_set(args.dataset_s, args.data_root, batch_size=args.batch_size, batch_size_test=128, 
            num_workers=args.num_workers, aux_data_filename=args.aux_data_filename, unsup_fraction=args.unsup_fraction, use_augmentation='randaugment', args=args)
            args.dataset = "cifar10"
            return dataset_train, dataset_test


def build_dataset_pre(args):

    if args.dataset=='imagenet':
        # simple augmentation
        if args.use_normalize:
            transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)])
        else:
            transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()])
        dataset_train = torchvision.datasets.ImageFolder(os.path.join(args.data_root, 'train'), transform=transform_train)
    elif args.use_normalize==False and args.dataset=='cifar10' and args.use_edm:
        args.dataset = args.dataset + 's'
        dataset_train, dataset_test = load_set(args.dataset, args.data_root, batch_size=args.batch_size, batch_size_test=128, 
        num_workers=args.num_workers, aux_data_filename=args.aux_data_filename, unsup_fraction=args.unsup_fraction)
        return dataset_train, dataset_test
    elif args.use_normalize and args.dataset=='cifar10' and args.use_edm:
        print('edm data should not use normalization.')
        exit(0)
    elif args.dataset=='cifar10' and not args.use_edm:
        # simple augmentation
        if args.use_normalize:
            transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=cifar10_mean, std=cifar10_std)])
        else:
            transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()])
        dataset_train = torchvision.datasets.CIFAR10(root=args.data_root, transform=transform_train, download=True, train=True)

    return dataset_train



