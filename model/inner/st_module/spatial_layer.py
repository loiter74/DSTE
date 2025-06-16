import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from model.inner.st_module.net_module import Conv1d_with_init


def default(val, default_val):
    return val if val is not None else default_val

def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

class Attn_spa(nn.Module):
    def __init__(self, dim, seq_len, k=256, heads=8, dim_head=None, one_kv_head=False, share_kv=False, dropout=0.):
        super().__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by the number of heads'

        self.seq_len = seq_len
        self.k = k

        self.heads = heads

        dim_head = default(dim_head, dim // heads)
        self.dim_head = dim_head

        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)

        kv_dim = dim_head if one_kv_head else (dim_head * heads)
        self.to_k = nn.Linear(dim, kv_dim, bias=False)
        self.proj_k = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.share_kv = share_kv
        if not share_kv:
            self.to_v = nn.Linear(dim, kv_dim, bias=False)
            self.proj_v = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(dim_head * heads, dim)

    def forward(self, x, itp_x=None, **kwargs):
        b, n, d, d_h, h, k = *x.shape, self.dim_head, self.heads, self.k

        v_len = n if itp_x is None else itp_x.shape[1]
        assert v_len == self.seq_len, f'the sequence length of the values must be {self.seq_len} - {v_len} given'

        q_input = x if itp_x is None else itp_x
        queries = self.to_q(q_input)
        proj_seq_len = lambda args: torch.einsum('bnd,nk->bkd', *args)

        k_input = x if itp_x is None else itp_x
        v_input = x

        keys = self.to_k(k_input)
        values = self.to_v(v_input) if not self.share_kv else keys
        kv_projs = (self.proj_k, self.proj_v if not self.share_kv else self.proj_k)

        # project keys and values along the sequence length dimension to k
        keys, values = map(proj_seq_len, zip((keys, values), kv_projs))

        # merge head into batch for queries and key / values
        queries = queries.reshape(b, n, h, -1).transpose(1, 2)

        merge_key_values = lambda t: t.reshape(b, k, -1, d_h).transpose(1, 2).expand(-1, h, -1, -1)
        keys, values = map(merge_key_values, (keys, values))

        # attention
        dots = torch.einsum('bhnd,bhkd->bhnk', queries, keys) * (d_h ** -0.5)
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhnk,bhkd->bhnd', attn, values)

        # split heads
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


class SpaLearning(nn.Module):
    def __init__(self, channels, nheads, target_dim, proj_t):
        super().__init__()
        # target_dim -> K
        self.attn = Attn_spa(dim=channels, seq_len=target_dim, k=proj_t, heads=nheads)
        self.cond_proj = Conv1d_with_init(2 * channels, channels, 1)
        self.norm1_local = nn.GroupNorm(4, channels)
        self.norm1_attn = nn.GroupNorm(4, channels)
        self.ff_linear1 = nn.Linear(channels, channels * 2)
        self.ff_linear2 = nn.Linear(channels * 2, channels)
        self.norm2 = nn.GroupNorm(4, channels)

    def forward(self, y, itp_y=None):
        B, channel, K, L = y.shape
        y_local = y

        y_attn = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        if itp_y is not None:
            itp_y_attn = itp_y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
            y_attn = self.attn(y_attn.permute(0, 2, 1), itp_y_attn.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y_attn = self.attn(y_attn.permute(0, 2, 1)).permute(0, 2, 1)
        y_attn = y_attn.reshape(B, L, channel, K).permute(0, 2, 3, 1)

        y_attn = self.norm1_attn(y_attn)
        y_in2 = y_local + y_attn

        y = F.relu(self.ff_linear1(y_in2.reshape(-1, channel)))
        y = self.ff_linear2(y).reshape(B, channel, K, L)
        y = y + y_in2

        y = self.norm2(y)
        return y
