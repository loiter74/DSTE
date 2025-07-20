import torch
from torch import nn


class MultiTConv1d(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size=1,
                 dropout=0.1, activ=None, num_layers=4, channel_reduction='half'):
        super().__init__()
        self.input_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.output_proj = TConv1d(hidden_channels//num_layers*num_layers, out_channels, kernel_size=1, dilation=1)

        self.layers = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.norm2 = nn.ModuleList()
        self.activ = activ

        for i in range(num_layers):
            # 确定当前层的输出通道
            layer_channels = hidden_channels//num_layers

            # 添加卷积层
            conv = TConv1d(
                in_channels=in_channels if i == 0 else layer_channels,
                out_channels=layer_channels,
                kernel_size=3, #2**(i+1)+1,
                dropout=dropout if i < num_layers - 1 else 0.0,  # 最后一层不加dropout
                dilation=2**i,
            )

            # 构建顺序模块
            self.layers.append(nn.Sequential(conv))
            self.norm.append(nn.BatchNorm2d(layer_channels))
            self.norm2.append(nn.BatchNorm1d(layer_channels))

            # 添加激活函数
        if activ == "gelu":
            self.activation = nn.GELU()
        elif activ == "relu":
            self.activation = nn.ReLU()

    def forward(self, x):

        total_x = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if len(x.shape) == 4:
                x = self.norm[i](x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
            if len(x.shape) == 3:
                x = self.norm2[i](x)
            total_x += [x]
        if self.activ is not None:
            x = self.activation(x)

        out = torch.concat(total_x, dim=1 if len(x.shape) == 3 else 2)
        x = self.output_proj(out)
        return x


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
