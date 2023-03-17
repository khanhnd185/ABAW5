import torch
import numpy as np 
from math import cos, pi
from sklearn.metrics import f1_score

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

def ex_metric(y, yhat):
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