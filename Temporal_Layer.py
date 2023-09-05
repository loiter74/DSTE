# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 14:50:13 2023

@author: dell
"""
import torch
import torch.nn as nn

class CT_MSA(nn.Module):
    # Causal Temporal MSA
    def __init__(self,
                 n_feat,
                 num_time,  # the number of time slot
                 mlp_dim=32,  # mlp dimension
                 window_size=24,  # the size of local window
                 depth=2,  # the number of MSA in CT-MSA
                 n_heads=2,  # the number of heads
                 dropout=0.,  # dropout rate
                 device=None):  # device, e.g., cuda
        super().__init__()
        
        n_hid = n_feat
        self.pos_embedding = nn.Parameter(torch.randn(1, num_time, n_hid))
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                TemporalAttention(dim=n_hid, n_heads=n_heads, window_size=window_size, dropout=dropout, device=device),
                PreNorm(n_hid, FeedForward(n_hid, mlp_dim, dropout=dropout))]))

    def forward(self, x):
        # x: [b, n, t, c]
        b, n, t, c = x.shape
        x = x.reshape(b*n, t, c)  # [b*n, t, c]
        x = x + self.pos_embedding  # [b*n, t, c]
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        x = x.reshape(b, n, t, c)
        return x

class TemporalAttention(nn.Module):
    def __init__(self, dim, n_heads=2, window_size=0, qkv_bias=False, qk_scale=None, dropout=0., causal=True, device=None):
        super().__init__()
        assert dim % n_heads == 0, f"dim {dim} should be divided by num_heads {n_heads}."

        self.dim = dim
        self.num_heads = n_heads
        self.causal = causal
        head_dim = dim // n_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.window_size = window_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        self.mask = torch.tril(torch.ones(window_size, window_size)).to(
            device)  # mask for causality

    def forward(self, x):
        B_prev, T_prev, C_prev = x.shape
        if self.window_size > 0:
            x = x.reshape(-1, self.window_size, C_prev)  # create local windows
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # merge key padding and attention masks
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [b, heads, T, T]
        if self.causal:
            attn = attn.masked_fill_(self.mask == 0, float("-inf"))
        x = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if self.window_size > 0:  # reshape to the original size
            x = x.reshape(B_prev, T_prev, C_prev)
        return x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
    
if __name__ == "__main__":
    import torch.optim as optim
    from torch.utils.data import DataLoader
    # 定义超参数
    batch_size = 16
    learning_rate = 1e-4
    num_epochs = 1000

    num_samples = 100
    data_shape = (10, 120, 16)
    dummy_data = torch.ones(num_samples, *data_shape)
    

    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(dummy_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 创建模型
    model = CT_MSA(n_feat=16, num_time=120) # num_time的意义是时间编码

    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 使用均方误差作为示例损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for batch_data in train_loader:
            inputs = batch_data[0]  # 获取输入数据
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)  # 使用均方误差计算损失
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        average_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {average_loss:.4f}")

    # 保存模型
    # torch.save(model.state_dict(), 'ds_msa_model.pth')
