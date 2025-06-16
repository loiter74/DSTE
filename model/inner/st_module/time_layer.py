import torch
import torch.nn as nn

from model.inner.st_module.net_module import Attn_tem, Conv1d_with_init


class TConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.1):
        super(TConv1d, self).__init__()
        self.padding = nn.ConstantPad1d(((kernel_size - 1) * dilation, 0), 0)
        self.convolution = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if len(x.shape) == 4:
            b, n = x.shape[:2]
            x = x.flatten(0, 1)
            y = self.convolution(self.padding(x))
            y = y.reshape([b, n, -1, y.shape[-1]])
        else:
            # (batch_size, in_channels, sequence_length)
            x = self.padding(x)
            y = self.convolution(x)
        y = self.dropout(torch.relu(y))
        return y


class LSTM(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=out_channels, num_layers=2, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if len(x.shape) == 4:
            b, n, c, t = x.shape
            x = x.flatten(0, 1).permute(0, 2, 1)
            y, _ = self.lstm(x)
            y = y.permute(0, 2, 1)
            y = y.reshape([b, n, -1, y.shape[-1]])
        else:
            y, _ = self.lstm(x.permute(0, 2, 1))
            y = y.permute(0, 2, 1)
        y = self.dropout(torch.relu(y))
        return y

class TemporalLearning(nn.Module):
    def __init__(self, in_channels, channels, out_channels, tcn_kernel_size, tcn_dilation, dropout=0.1, nheads=4):
        super().__init__()
        self.input_proj = nn.Conv1d(in_channels=in_channels, out_channels=channels, kernel_size=tcn_kernel_size)
        self.lstm1 = nn.LSTM(input_size=channels, hidden_size=channels//2, num_layers=4, batch_first=True, bidirectional=True)
        #self.lstm2 = nn.LSTM(input_size=channels, hidden_size=channels, num_layers=2, batch_first=True)

        self.tcn = TConv1d(channels, channels, kernel_size=tcn_kernel_size, dilation=tcn_dilation)
        self.time_attn = Attn_tem(heads=nheads, layers=4, channels=channels)

        self.cond_proj = Conv1d_with_init(2 * channels, out_channels, 1)

        #self.relu = nn.ReLU()
        self.group_norm = nn.GroupNorm(4, out_channels)
        self.out_channels = out_channels
        self.dropout = nn.Dropout(dropout)

    def forward(self, y, itp_y=None):
        B, K, channel, L = y.shape
        if L == 1: return y


        y = self.input_proj(y)

        y_ori = y.clone()
        y_ori = y_ori.permute(0, 2, 1, 3)

        #y = self.tcn(y)
        y = y.reshape(B * K, channel, L)

        v, _ = self.lstm1(y.permute(2, 0, 1))
        #v = y.permute(2, 0, 1)
        if itp_y is None: # self attn
            itp_y = y.clone()
        itp_y = itp_y.reshape(B * K, channel, L)

        q, _ = self.lstm2(itp_y.permute(2, 0, 1))
        #q = itp_y.permute(2, 0, 1)

        y = self.time_attn(q, q, v)
        # y, _ = self.lstm2(y)
        y = y.permute(1, 2, 0).reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K , L)

        y = (self.cond_proj(torch.cat([y.reshape(B*K, channel, L), y_ori.reshape(B*K, channel, L)], dim=1)))
        y = y.reshape(B, self.out_channels, K , L)
        #y = self.group_norm(y)

        y = self.dropout(y)
        y = y.permute(0, 2, 1, 3)
        return y




