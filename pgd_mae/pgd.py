import torch
import torch.nn as nn
from timm.loss import SoftTargetCrossEntropy

from .attack import Attack


class PGD(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=10, random_start=True)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, device=None, eps=8/255, alpha=2/255, steps=10, random_start=True, upper_limit=1, lower_limit=0):
        super().__init__('PGD', model, device)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ['default', 'targeted']
        self.upper_limit = upper_limit
        self.lower_limit = lower_limit
        self.mixup = False

    def clamp(self, X, upper_limit, lower_limit):
        return torch.max(torch.min(X, upper_limit), lower_limit)

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        if self.mixup:
            loss = SoftTargetCrossEntropy()
        else:
            loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            if torch.is_tensor(self.eps):
                for i in range(len(self.eps)):
                    adv_images[:,i,:,:] = adv_images[:,i,:,:] + \
                        torch.empty_like(adv_images[:,i,:,:]).uniform_(-self.eps[i][0][0].item(), self.eps[i][0][0].item())
            else:
                adv_images = adv_images + \
                    torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = self.clamp(adv_images, self.upper_limit, self.lower_limit).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)

            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = self.clamp(adv_images - images, -self.eps, self.eps)
            adv_images = self.clamp(images + delta, self.upper_limit, self.lower_limit).detach()

        return adv_images