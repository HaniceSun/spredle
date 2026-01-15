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
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true, epsilon=1e-10):
        s1 = y_true[:, 0, :] * torch.log(y_pred[:, 0, :] + epsilon)
        s2 = y_true[:, 1, :] * torch.log(y_pred[:, 1, :] + epsilon)
        s3 = y_true[:, 2, :] * torch.log(y_pred[:, 2, :] + epsilon)
        s = s1 + s2 + s3
        return(-torch.mean(s))

class CustomLossClsReg(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true, epsilon=1e-10):
        s1 = y_true[:, 0, :] * torch.log(y_pred[:, 0, :] + epsilon)
        s2 = y_true[:, 1, :] * torch.log(y_pred[:, 1, :] + epsilon)
        s3 = y_true[:, 2, :] * torch.log(y_pred[:, 2, :] + epsilon)
        s4 = - ((y_true[:, 3, :] - y_pred[:, 3, :]) ** 2)
        s = s1 + s2 + s3 + s4
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
