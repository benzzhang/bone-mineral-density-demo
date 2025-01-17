# coding: utf8
'''
Author: Pengbo
Date: 2022-05-15 16:17:31
Description: loss function

'''
import numpy as np
import torch
from torch.nn import Module
from torch import nn
from torch.nn import functional as F
from medpy.metric.binary import dc

__all__ = ['Segloss', 'DiceCoeff', 'L2Loss', 'BCELoss', 'MaskBCELoss', 'MSELoss', 'DiceLoss', 'BCEFocalLoss', 'TorchBCELoss', 'FocalLoss', 'ConfidentMSELoss']

class Segloss(nn.Module):
    def __init__(self):
        super(Segloss, self).__init__()

    def forward(self, logits, target):
        FP = torch.sum((1 - target) * logits)
        FN = torch.sum((1 - logits) * target)
        return FP, FN


class DiceCoeff(nn.Module):
    def __init__(self):
        super(DiceCoeff, self).__init__()

    def forward(self, logits, target):
        logits = logits.cpu().detach().numpy()
        # print('logits:', logits.shape)
        # print(logits)
        target = target.cpu().detach().numpy()
        # print('target:', target.shape)
        # print(target)

        # 返回的dc()没有用eps，所以mask无阳性目标时，两者取反：
        # 1.logits也无阳性目标，防止分母为零，返回1
        # 2.logits有假阳目标，Dice很低
        if np.count_nonzero(target) == 0:
            target = target + 1
            logits = 1 - logits
        
        # 'value > 0' will be considered as True, 'value = 0' will be considered as False in func 'dc()'
        return dc(logits, target)


class DiceLoss(nn.Module):
    """Dice loss.

    :param input: The input (predicted)
    :param target:  The target (ground truth)
    :returns: the Dice score between 0 and 1.
    """
    def __init__(self, eps = 1e-5):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, logits, target):
        lflat = logits.view(-1)
        tflat = target.view(-1)

        # if tflat.sum() == 0:
        #     tflat = tflat + 1
        #     lflat = 1 - lflat

        intersection = (lflat * tflat).sum()
        union = lflat.sum() + tflat.sum()
        
        dice = (2.0 * intersection + self.eps) / (union + self.eps)
        return 1. - dice, intersection.item(), union.item()


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, logits, target):
        mse = torch.nn.MSELoss(reduction='mean')
        loss = mse(logits, target) / 26 * 1000
        return loss


class L2Loss(nn.Module):
    def __init__(self, div_element = False):
        super(L2Loss, self).__init__()
        self.div_element = div_element

    def forward(self, output, target):
        loss = torch.sum(torch.pow(output-target, 2) )
        if self.div_element:
            loss = loss / output.numel()
        else:
            loss = loss / output.size(0) / 2
        return loss


class TorchBCELoss(nn.Module):
    def __init__(self, reduction='sum'):
        super(TorchBCELoss, self).__init__()
        self.loss_fun = torch.nn.BCELoss(reduction=reduction)

    def forward(self, logits, target, mask=None):
        loss = self.loss_fun(logits, target)
        return loss


# from Github
class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, logits, target):
        return self.bce_loss(logits, target)

# class BCELoss(nn.Module):
#     def __init__(self):
#         super(BCELoss, self).__init__()
#         self.pos_weight = 1
#         self.eps = 1e-6
#         self.reduction = 'sum'

#     def forward(self, logits, target):
#         # logits: [N, *], target: [N, *]
#         loss = - self.pos_weight * target * torch.log(logits+self.eps) - \
#                (1 - target) * torch.log(1 - logits - self.eps)
#         if self.reduction == 'mean':
#             loss = loss.mean()
#         elif self.reduction == 'sum':
#             loss = loss.sum()
#         return loss


class MaskBCELoss(nn.Module):
    def __init__(self, reduction='sum'):
        super(MaskBCELoss, self).__init__()
        self.pos_weight = 1
        self.eps = 1e-6
        self.reduction = reduction

    def forward(self, logits, target, mask):
        # logits: [N, *], target: [N, *]
        loss = - self.pos_weight * target * torch.log(logits+self.eps) - \
               (1 - target) * torch.log(1 - logits - self.eps)

        if self.reduction == 'mean':
            loss = torch.mean(torch.mean(loss, dim=(2,3)) * mask)
        elif self.reduction == 'sum':
            loss = torch.sum(torch.sum(loss, dim=(2,3)) * mask)
        elif self.reduction == 'keep':
            loss = torch.sum(torch.sum(loss, dim=(2,3)) * mask, dim=0) / torch.sum(mask, dim=0)
            #loss = torch.sum(torch.sum(loss, dim=(2,3)) * mask, dim=0)

        return loss


class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=0.6, gamma=0.2, num_classes = 2):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.num_classes = 2
        self.reduction = 'mean'

    def forward(self, logits, target):
        alpha = self.alpha
        gamma = self.gamma
        loss = - alpha * (1 - logits) ** gamma * target * torch.log(logits) - \
               (1 - alpha) * logits ** gamma * (1 - target) * torch.log(1 - logits)
        return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes = 2, size_average=True):
        """
        focal_loss, -α(1-yi)**γ *ce_loss(xi,yi)
        :param alpha: loss weight for each class. 
        :param gamma: hyper-parameter to adjust hard sample
        :param num_classes:
        :param size_average:
        """
        super(FocalLoss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1 
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # set alpha to [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
        self.gamma = gamma

    def forward(self, preds, labels):
        """
        :param preds:   prediction. the size: [B,N,C] or [B,C]
        :param labels:  ground-truth. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1,preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1) # log_softmax
        preds_softmax = torch.exp(preds_logsoft)    # softmax

        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))  
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        self.alpha = self.alpha.gather(0,labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft) 

        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class ConfidentMSELoss(Module):
    def __init__(self, threshold=0.96):
        self.threshold = threshold
        super().__init__()

    def forward(self, input, target):
        n = input.size(0)
        conf_mask = torch.gt(target, self.threshold).float()
        input_flat = input.view(n, -1)
        target_flat = target.view(n, -1)
        conf_mask_flat = conf_mask.view(n, -1)
        diff = (input_flat - target_flat)**2
        diff_conf = diff * conf_mask_flat
        loss = diff_conf.mean()
        return loss
