from math import cos, pi
from re import S
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch 
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        n = float(n)
        self.sum += val * n
        self.count += n
    
    def avg(self):
        return (self.sum / self.count)

def adjust_learning_rate(optimizer, epoch, epochs, init_lr, iteration, num_iter):

    current_iter = iteration + epoch * num_iter
    max_iter = epochs * num_iter
    lr = init_lr * (1 + cos(pi * current_iter / max_iter)) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def load_state_dict(model,path):
    checkpoints = torch.load(path,map_location=torch.device('cpu'))
    state_dict = checkpoints['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module.' in k:
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    # load params
    model.load_state_dict(new_state_dict,strict=False)
    return model

EPS = 1e-8

class MaskedCELoss(nn.Module):
    def __init__(self, weight=None, ignore_index=-1):
        super(MaskedCELoss, self).__init__() 
        self.ce = nn.CrossEntropyLoss(reduction='none', weight=weight, ignore_index=ignore_index)
    
    def forward(self, x, y, mask):
        loss = self.ce(x, y)
        loss = loss.mean(dim=-1)
        loss = loss * mask
        return loss.sum() / (mask.sum() + EPS)


def au_metric(y, yhat, thresh=0.5):
    yhat = (yhat >= thresh)
    N, label_size = y.shape
    f1s = []
    for i in range(label_size):
        f1 = f1_score(y[:, i], yhat[:, i])
        f1s.append(f1)
    return np.mean(f1s), f1s

class WeightedAsymmetricLoss(nn.Module):
    def __init__(self, eps=1e-8, disable_torch_grad=True, weight=None):
        super(WeightedAsymmetricLoss, self).__init__()
        self.disable_torch_grad = disable_torch_grad
        self.eps = eps
        self.weight = weight

    def forward(self, x, y, mask):
        xs_pos = x
        xs_neg = 1 - x
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))

        if self.disable_torch_grad:
            torch.set_grad_enabled(False)
        neg_weight = 1 - xs_neg
        if self.disable_torch_grad:
            torch.set_grad_enabled(True)
        loss = los_pos + neg_weight * los_neg

        if self.weight is not None:
            loss = loss * self.weight.view(1,-1)

        loss = loss.mean(dim=-1)
        loss = loss * mask
        return -loss.sum() / (mask.sum() + EPS)
