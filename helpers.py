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

def EX_metric(y, yhat):
    i = np.argmax(yhat, axis=1)
    yhat = np.zeros(yhat.shape)
    yhat[np.arange(len(i)), i] = 1

    if not len(y.shape) == 1:
        if y.shape[1] == 1:
            y = y.reshape(-1)
        else:
            y = np.argmax(y, axis=-1)
    if not len(yhat.shape) == 1:
        if yhat.shape[1] == 1:
            yhat = yhat.reshape(-1)
        else:
            yhat = np.argmax(yhat, axis=-1)

    return f1_score(y, yhat, average='macro')

def CCC_score(x, y):
    vx = x - np.mean(x)
    vy = y - np.mean(y)
    rho = np.sum(vx * vy) / (np.sqrt(np.sum(vx**2)) * np.sqrt(np.sum(vy**2)))
    x_m = np.mean(x)
    y_m = np.mean(y)
    x_s = np.std(x)
    y_s = np.std(y)
    ccc = 2*rho*x_s*y_s/(x_s**2 + y_s**2 + (x_m - y_m)**2)
    return ccc

def va_metric(y, yhat):
    cccs = (float(CCC_score(y[:,0], yhat[:,0])), float(CCC_score(y[:,1], yhat[:,1])))
    avg_ccc = float(CCC_score(y[:,0], yhat[:,0]) + CCC_score(y[:,1], yhat[:,1])) / 2
    return avg_ccc, cccs

class RegressionLoss(nn.Module):
    def __init__(self):
        super(RegressionLoss, self).__init__() 
        self.loss = MaskNegativeCCCLoss()

    def forward(self, x, y, mask):
        loss1 = self.loss(x[:, 0], y[:, 0], mask) + self.loss(x[:, 1], y[:, 1], mask)
        return loss1

class MaskNegativeCCCLoss(nn.Module):
    def __init__(self):
        super(MaskNegativeCCCLoss, self).__init__()
    def forward(self, x, y, m):
        y = y.view(-1)
        x = x.view(-1)
        x = x * m
        y = y * m
        N = torch.sum(m)
        x_m = torch.sum(x) / N
        y_m = torch.sum(y) / N
        vx = (x - x_m) * m
        vy = (y - y_m) * m
        ccc = 2*torch.dot(vx, vy) / (torch.dot(vx, vx) + torch.dot(vy, vy) + N * torch.pow(x_m - y_m, 2) + EPS)
        return 1-ccc

