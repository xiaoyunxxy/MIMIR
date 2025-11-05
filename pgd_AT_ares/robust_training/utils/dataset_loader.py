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

from edm_data import load_data, load_set

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)


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
    elif args.dataset == "tiny-imagenet":
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


    # ------ test transform for imagenet
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
    

    if not args.use_normalize:
        if args.dataset=='imagenet':
            if args.aa!='noaug':
                # data augmentation transform
                print('use auto_augment: ', args.aa)
                mean = (0.0, 0.0, 0.0)
                std = (1.0, 1.0, 1.0)
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
                transform_train = transforms.Compose([
                transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()])

            transform_test = transforms.Compose(t)
        else:
            transform_train = transforms.Compose(
                [transforms.RandomCrop(32, padding=4),
                # transforms.Resize(224),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor()])

            transform_test = transforms.Compose([
                # transforms.Resize(224),
                transforms.ToTensor()])
    else:
        # use normalization with mean and std
        if args.dataset=="imagenet" or args.dataset=="imagenette" or args.dataset=="tiny-imagenet":
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        elif args.dataset=="cifar10" or args.dataset=="cifar10s":
            mean = cifar10_mean
            std = cifar10_std
        else:
            print('check dataset for mean and std!')
            exit(0)

        t.append(transforms.Normalize(mean, std))

        if args.aa!='noaug':
            # data augmentation transform
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
            transform_test = transforms.Compose(t)
        else:
            if args.dataset=='imagenet' or args.dataset=='imagenette':
                transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)])

                transform_test = transforms.Compose(t)
            else: 
                transform_train = transforms.Compose(
                    [transforms.RandomCrop(args.input_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)])

                # transform_test = transforms.Compose([
                #     transforms.ToTensor(),
                #     transforms.Normalize(mean, std)])

                # for transfer 
                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)])

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
        elif args.dataset == "tiny-imagenet":
            return torchvision.datasets.ImageFolder(root=args.data_root+'/tiny-imagenet-200/train' if is_train \
                                    else args.data_root + '/tiny-imagenet-200/val', transform=transform)
        elif args.dataset == "imagenet":
            return torchvision.datasets.ImageFolder(root=args.data_root+'/train' if is_train \
                                    else args.data_root + '/val', transform=transform)
        elif args.dataset == "imagenette":
            return torchvision.datasets.ImageFolder(root=args.data_root+'/imagenette2/train' if is_train \
                                    else args.data_root + '/imagenette2/val', transform=transform)
        elif args.dataset == "cifar10s":
            dataset_train, dataset_test = load_set(args.dataset_s, args.data_root, batch_size=args.batch_size, batch_size_test=128, 
            num_workers=args.num_workers, aux_data_filename=args.aux_data_filename, unsup_fraction=args.unsup_fraction, use_augmentation=args.edm_aug, args=args)
            args.dataset = "cifar10"
            return dataset_train, dataset_test
        elif args.dataset == "tiny-imagenets":
            dataset_train, dataset_test = load_set(args.dataset_s, args.data_root+'/tiny-imagenet-200/', batch_size=args.batch_size, batch_size_test=128, 
            num_workers=args.num_workers, aux_data_filename=args.aux_data_filename, unsup_fraction=args.unsup_fraction, use_augmentation=args.edm_aug, args=args)
            args.dataset = "tiny-imagenet"
            return dataset_train, dataset_test
