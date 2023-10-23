import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from autoattack import AutoAttack
import torchattacks
import util.misc as misc
from timm.utils import accuracy
import PIL

# installing AutoAttack by: pip install git+https://github.com/fra31/auto-attack


cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)

def no_nor_loader(args):
    if args.dataset=="cifar10":
        test_transform_nonorm = transforms.Compose([
            transforms.ToTensor()
        ])
        test_dataset_nonorm = datasets.CIFAR10(args.data_root, 
            train=False, transform=test_transform_nonorm, download=True)
    if args.dataset=="imagenette" or args.dataset=="imagenet" :
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

        test_transform_nonorm = transforms.Compose(t)
        test_dataset_nonorm = datasets.ImageFolder(args.data_root+'/val',test_transform_nonorm)

    test_loader_nonorm = torch.utils.data.DataLoader(
        dataset=test_dataset_nonorm,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=16)

    return test_loader_nonorm

def normalize(data_set, X):
    if data_set=="cifar10":
        mu = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
        std = torch.tensor(cifar10_std).view(3, 1, 1).cuda()
    elif data_set=="imagenette" or data_set=="imagenet" :
        mu = torch.tensor(imagenet_mean).view(3, 1, 1).cuda()
        std = torch.tensor(imagenet_std).view(3, 1, 1).cuda()
    return (X - mu) / std

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

def evaluate_aa(args, model, log_path):
    test_loader_nonorm = no_nor_loader(args)
    model.eval()
    l = [x for (x, y) in test_loader_nonorm]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_loader_nonorm]
    y_test = torch.cat(l, 0)

    if args.use_normalize:
        model = normalize_model(model, args.dataset)
    adversary = AutoAttack(model, norm='Linf', eps=args.eps, version='standard',log_path=log_path)
    X_adv = adversary.run_standard_evaluation(x_test, y_test, bs=args.batch_size)


def evaluate_pgd(args, model, device, eval_steps=10):
    test_loader_nonorm = no_nor_loader(args)
    if args.use_normalize:
        model = normalize_model(model, args.dataset)

    attack = torchattacks.PGD(model, eps=args.eps,
                            alpha=args.alpha, steps=eval_steps, random_start=True)
    header='PGD {} Test:'.format(eval_steps)
    test_stats = evaluate_adv(attack, test_loader_nonorm, model, device, header)
    print(f"Accuracy of the network on the {len(test_loader_nonorm)} test images: {test_stats['acc1']:.1f}%")

def evaluate_cw(args, model, device, eval_steps=20):
    test_loader_nonorm = no_nor_loader(args)
    if args.use_normalize:
        model = normalize_model(model, args.dataset)

    attack = torchattacks.CW(model=model, lr=0.05, steps=eval_steps)
    attack.device = device
    header='CW {} Test:'.format(eval_steps)
    test_stats = evaluate_adv(attack, test_loader_nonorm, model, device, header)
    print(f"Accuracy of the network on the {len(test_loader_nonorm)} test images: {test_stats['acc1']:.1f}%")

def evaluate_adv(attack, data_loader, model, device, header='ADV Test:'):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
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

