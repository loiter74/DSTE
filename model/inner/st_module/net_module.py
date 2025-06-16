import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
#from linear_attention_transformer import LinearAttentionTransformer


class GatedFusion(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        layer = Conv1d_with_init(2*in_channels, out_channels)
        self.gate = nn.Sequential(
            layer,
            nn.BatchNorm1d(out_channels),
            nn.Sigmoid(),
            nn.Dropout(0.01),
        )

    def forward(self, x, y):
        b, n, c, t = x.shape
        x = x.reshape(b*n, c, t)
        y = y.reshape(b*n, c, t)

        input_combined = torch.cat([x, y], dim=1)
        gate = self.gate(input_combined)
        output = gate * x + (1 - gate) * y
        output = output.reshape(b, n, c, t)

        return output

def Attn_tem(heads=8, layers=1, channels=64):
    encoder_layer = TransformerEncoderLayer_QKV(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return TransformerEncoder_QKV(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels, out_channels, kernel_size=1, dropout=0.1, activ=None):
    # 废弃方法 有问题
    layer = Conv1d(in_channels, out_channels, kernel_size, dropout=0.1)

    if dropout > 0:
         layer = nn.Sequential(layer, nn.Dropout(dropout))
    if activ== "gelu":
        return nn.Sequential(layer, nn.GELU())
    elif activ == "relu":
        return nn.Sequential(layer, nn.ReLU())
    return layer

class MultiConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1,
                 dropout=0.1, activ=None, num_layers=3, channel_reduction='half'):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norm = nn.ModuleList()
        current_channels = in_channels
        self.activ = activ
        # 通道变化策略
        reducers = {
            'half': lambda x: max(x // 2, 4),
            'double': lambda x: x * 2,
            'same': lambda x: x
        }
        channel_func = reducers.get(channel_reduction, 'same')
        self.out_proj = Conv1d(out_channels, out_channels, kernel_size=1)
        for i in range(num_layers):
            # 确定当前层的输出通道
            layer_out = out_channels if i == num_layers - 1 else channel_func(current_channels)

            # 添加卷积层
            conv = Conv1d(
                in_channels=current_channels,
                out_channels=layer_out,
                kernel_size=kernel_size,
                dropout=dropout if i < num_layers - 1 else 0.0  # 最后一层不加dropout
            )
            # 构建顺序模块
            self.layers.append(nn.Sequential(conv))
            self.norm.append(nn.BatchNorm2d(layer_out))
            current_channels = layer_out
            # 添加激活函数
        if activ == "gelu":
            self.activation = nn.GELU()
        elif activ == "relu":
            self.activation = nn.ReLU()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.norm[i](x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        if self.activ is not None:
            x = self.activation(x)
        # x = self.out_proj(x)
        return x


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

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerEncoderLayer_QKV(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="gelu"):
        super(TransformerEncoderLayer_QKV, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout) #Q(L, N, E) K(S, N, E) V(S, N, E)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer_QKV, self).__setstate__(state)

    def forward(self, query, key, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(query, key, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerEncoder_QKV(nn.Module):
    __constants__ = ['norm']
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder_QKV, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, query, key, src, mask=None, src_key_padding_mask=None):
        output = src
        for mod in self.layers:
            output = mod(query, key, output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

# def get_linear_trans(heads=8,layers=1,channels=64,localheads=0,localwindow=0):
#
#   return LinearAttentionTransformer(
#         dim = channels,
#         depth = layers,
#         heads = heads,
#         max_seq_len = 256,
#         n_local_attn_heads = 0,
#         local_attn_window_size = 0,
#     )
