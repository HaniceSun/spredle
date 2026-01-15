import sys
import numpy as np
import torch
torch.manual_seed(42)
import matplotlib
matplotlib.use('Agg')
import pylab as plt
import seaborn as sns
from importlib import resources

hyena_dir = f'{resources.files("spredle").parent.parent}/vendor/hyena'
sys.path.append(hyena_dir)
from hyena import *

class CustomLoss(torch.nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, y_pred, y_true, epsilon=1e-10):
        L = []
        for n in range(self.n_classes):
            s = y_true[:, n, :] * torch.log(y_pred[:, n, :] + epsilon)
            L.append(s)
        s = torch.stack(L, dim=0).sum(dim=0)
        return(-torch.mean(s))

class CustomLossReg(torch.nn.Module):
    def __init__(self, n_regs=1):
        super().__init__()
        self.n_regs = n_regs

    def forward(self, y_pred, y_true):
        L = []
        for n in range(self.n_regs):
            s = (y_true[:, n, :] - y_pred[:, n, :]) ** 2
            L.append(s)
        s = torch.stack(L, dim=0).sum(dim=0)
        return(torch.mean(s))

class CustomLossClsReg(torch.nn.Module):
    def __init__(self, n_classes=3, n_regs=1):
        super().__init__()
        self.n_classes = n_classes
        self.n_regs = n_regs

    def forward(self, y_pred, y_true, epsilon=1e-10):
        for n in range(self.n_classes):
            s = y_true[:, n, :] * torch.log(y_pred[:, n, :] + epsilon)
            L.append(s)
        for n in range(self.n_regs):
            s_reg = - ((y_true[:, self.n_classes + n, :] - y_pred[:, self.n_classes + n, :]) ** 2)
            L.append(s_reg)
        s = torch.stack(L, dim=0).sum(dim=0)
        return(-torch.mean(s))

class Config:
    def __init__(self, config):
        for k, v in config.items():
            if k in ['NWD', 'learning_rate']:
                setattr(self, k, eval(v))
            else:
                setattr(self, k, v)

class EarlyStopping:
    def __init__(self, patience=3, delta=0):
        self.patience = patience
        self.delta = delta
        self.val_loss_min = np.inf
        self.counter = 0
        self.stopped = False
        self.best_epoch = 0

    def __call__(self, val_loss, epoch):
        if val_loss < self.val_loss_min:
            self.val_loss_min = val_loss
            self.counter = 0
            self.best_epoch = epoch
        elif val_loss > self.val_loss_min + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped = True
