#!/usr/bin/env python

# torch package
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms


# torchattacks toolbox
import torchattacks
from pgd_mae import pgd_mae

def attack_loader(args, net):
    # Gradient Clamping based Attack
    if args.attack == "fgsm":
        return torchattacks.FGSM(model=net, eps=args.eps)

    elif args.attack == "bim":
        return torchattacks.BIM(model=net, eps=args.eps, alpha=2/255)

    elif args.attack == "pgd":
        return torchattacks.PGD(model=net, eps=args.eps,
                                alpha=2/255, steps=args.steps, random_start=True)

    elif args.attack == "cw":
        return torchattacks.CW(model=net, c=0.1, lr=0.1, steps=args.cwsteps)

    elif args.attack == "auto":
        return torchattacks.APGD(model=net, eps=args.eps)

    elif args.attack == "fab":
        return torchattacks.FAB(model=net, eps=args.eps, n_classes=args.n_classes)

    elif args.attack == "nifgsm":
        return torchattacks.NIFGSM(model=net, eps=args.eps, alpha=2/255, steps=args.steps, decay=1.0)

    elif args.attack == "pgd_mae":
        return pgd_mae.PGD_MAE(model=net, eps=args.eps,
                                alpha=2/255, steps=args.steps, random_start=True)



def dataset_loader(args):

    args.mean=0.5
    args.std=0.25

    # Setting Dataset Required Parameters
    if args.dataset   == "svhn":
        args.n_classes = 10
        args.img_size  = 32
        args.channel   = 3
    elif args.dataset == "cifar10":
        args.n_classes = 10
        args.img_size  = 32
        args.channel   = 3
    elif args.dataset == "tiny":
        args.n_classes = 200
        args.img_size  = 64
        args.channel   = 3
    elif args.dataset == "cifar100":
        args.n_classes = 100
        args.img_size  = 32
        args.channel   = 3
    elif args.dataset == "imagenet":
        args.n_classes = 1000
        args.img_size  = 224
        args.channel   = 3
    elif args.dataset == "imagenette":
        args.n_classes = 10
        args.img_size  = 224
        args.channel   = 3

    transform_train = transforms.Compose(
        [transforms.RandomCrop(args.img_size, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()]
    )


    transform_test = transforms.Compose(
        [transforms.ToTensor()]
    )

    if args.dataset == "imagenet" or args.dataset == "imagenette":
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.img_size, scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        transform_test = transforms.Compose([
                transforms.Resize((args.img_size, args.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5])
            ])

    # Full Trainloader/Testloader
    trainloader = torch.utils.data.DataLoader(dataset(args, True,  transform_train), batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=32)
    testloader  = torch.utils.data.DataLoader(dataset(args, False, transform_test),  batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=32)

    return trainloader, testloader


def dataset(args, train, transform):
        if args.dataset == "cifar10":
            return torchvision.datasets.CIFAR10(root=args.data_root, transform=transform, download=True, train=train)
        elif args.dataset == "cifar100":
            return torchvision.datasets.CIFAR100(root=args.data_root, transform=transform, download=True, train=train)
        elif args.dataset == "svhn":
            return torchvision.datasets.SVHN(root=args.data_root, transform=transform, download=True,
                                    split="train" if train else "test")
        elif args.dataset == "tiny":
            return torchvision.datasets.ImageFolder(root=args.data_root+'/tiny-imagenet-200/train' if train \
                                    else args.data_root + '/tiny-imagenet-200/valid', transform=transform)
        elif args.dataset == "imagenet":
            return torchvision.datasets.ImageFolder(root=args.data_root+'/imagenet/train' if train \
                                    else args.data_root + '/imagenet/val', transform=transform)
        elif args.dataset == "imagenette":
            return torchvision.datasets.ImageFolder(root=args.data_root+'/imagenette2/train' if train \
                                    else args.data_root + '/imagenette2/val', transform=transform)