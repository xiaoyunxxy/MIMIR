import torch
import torch.nn as nn

from .attack import Attack


class PGD_MAE_FAST(Attack):
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

    def __init__(self, model, device=None, eps=8/255, alpha=2/255, steps=10, random_start=True):
        super().__init__('PGD_MAE_FAST', model, device)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + \
                torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for i in range(self.steps):
            adv_images.requires_grad = True

            if i==0:
                x, ids_restore, ids_keep, mask = self.prepare_encoder_random(adv_images)
                loss, _, _, _ = self.model.forward_for_pgd(x, ids_restore, mask, adv_images)
            else:
                x = self.prepare_encoder(adv_images, ids_keep)
                loss, _, _, _ = self.model.forward_for_pgd(x, ids_restore, mask, adv_images)

            # Update adversarial images
            grad = torch.autograd.grad(loss, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images,
                                min=-self.eps, max=self.eps)

            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images


    def prepare_encoder(self, x, ids_keep):
        # embed patches
        x = self.model.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.model.pos_embed[:, 1:, :]

        N, L, D = x.shape
        # masking with previous random mask
        x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # append cls token
        cls_token = self.model.cls_token + self.model.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        return x

    def prepare_encoder_random(self, x, mask_ratio = 0.75):
        # embed patches
        x = self.model.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.model.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore, ids_keep = self.model.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.model.cls_token + self.model.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        return x, ids_restore, ids_keep, mask
