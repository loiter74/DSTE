# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 09:18:22 2023

@author: dell
"""

import torch
import torch.nn as nn
from layers import GAT, CT_MSA, DiffusionEmbedding

class STD_Module(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len, st_blocks, diff_steps=1000, device=None):
        super(STD_Module, self).__init__()
      
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.st_blocks = st_blocks
        
        self.bn = nn.ModuleList()
        self.s_modules = nn.ModuleList()
        self.t_modules = nn.ModuleList()
        self.outfit_module = nn.Conv2d(in_channels=input_dim*st_blocks, out_channels=input_dim, kernel_size=(1, 1), stride=(1, 1))
        
        self.diffusion_embedding = DiffusionEmbedding(num_steps=diff_steps, projection_dim=input_dim)
      
        for b in range(self.st_blocks):
            window_size = self.seq_len // 2 ** (self.st_blocks - b - 1) # 时间因果卷积窗口
            self.t_modules.append(CT_MSA(n_feat=self.input_dim, window_size=window_size, num_time=self.seq_len, device=device)) 
            """魔改, 看来是没对接好"""
            self.s_modules.append(GAT(n_feat=self.hidden_dim, n_hid=self.hidden_dim))
            self.bn.append(nn.BatchNorm2d(self.input_dim))
            
    def forward(self, x, adj, diff_t):
        '''
        inputs: the historical data  [b, c, n, t]
        '''
        d = []  # deterministic states
        for i in range(self.st_blocks):
            B, C, N, T = x.shape 
            embed_t = self.diffusion_embedding(diff_t).view(1, -1, 1, 1)
            y = x + embed_t
            y = self.s_modules[i](y, adj)
            y = self.t_modules[i](y)  # [b, c, n, t]
            y = self.bn[i](y)
            d.append(y)
        d = torch.stack(d)  # [num_blocks, b, c, n, t]  d 中有三层尺度的数据
        num_blocks, B, C, N, T = d.shape
        y_hat = d.reshape(B, -1, N, T)  # [B, num_blocks*C, N, T]
        y_hat = self.outfit_module(y_hat)
       
        return y_hat

