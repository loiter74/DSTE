import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


def Conv1d_with_init(in_channels, out_channels, kernel_size=1, dropout=0.1, activ=None):
    layer = Conv1d(in_channels, out_channels, kernel_size, dropout=0.1)

    if dropout > 0:
         layer = nn.Sequential(layer, nn.Dropout(dropout))
    if activ== "gelu":
        return nn.Sequential(layer, nn.GELU())
    elif activ == "relu":
        return nn.Sequential(layer, nn.ReLU())
    return layer


class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, dropout=0.1):
        super(Conv1d, self).__init__()
        self.convolution = nn.Conv1d(in_channels, out_channels, kernel_size)
        nn.init.xavier_uniform_(self.convolution.weight, gain=1e-6)
       # nn.init.zeros_(self.bias)

    def forward(self, x):
        if len(x.shape) == 4:
            b, n = x.shape[:2]
            x = x.flatten(0, 1)
            y = self.convolution(x)
            y = y.reshape([b, n, -1, y.shape[-1]])
        else:
            # (batch_size, in_channels, sequence_length)
            y = self.convolution(x)
        return y

