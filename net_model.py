# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 09:18:22 2023

@author: dell
"""

import torch
import torch.nn as nn
from layers import GAT, CT_MSA

class ST_Module(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len, st_blocks):
        super(ST_Module, self).__init__()
      
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.st_blocks = st_blocks
        
        self.bn = nn.ModuleList()
        self.s_modules = nn.ModuleList()
        self.t_modules = nn.ModuleList()
        self.outfit_module = nn.Conv2d(in_channels=input_dim*st_blocks, out_channels=input_dim, kernel_size=(1, 1), stride=(1, 1))
        
        self.diff_models = nn.ModuleList()
      
        for b in range(self.st_blocks):
            window_size = self.seq_len // 2 ** (self.st_blocks - b - 1) # 时间因果卷积窗口
            self.t_modules.append(CT_MSA(n_feat=self.input_dim, window_size=window_size, num_time=self.seq_len)) 
            self.s_modules.append(GAT(n_feat=self.input_dim, n_hid=self.hidden_dim))
            self.bn.append(nn.BatchNorm2d(self.input_dim))
            
    def forward(self, x, adj):
        '''
        inputs: the historical data  [b, c, n, t]
        '''
        d = []  # deterministic states
        for i in range(self.st_blocks):
            B, C, N, T = x.shape 
            #embed_t = self.diff_models[i](diff_t).view(B, -1, 1, 1)
            #x = x + embed_t
            x = self.s_modules[i](x, adj)
            x = self.t_modules[i](x)  # [b, c, n, t]
            x = self.bn[i](x)
            d.append(x)
        d = torch.stack(d)  # [num_blocks, b, c, n, t]  d 中有三层尺度的数据
        num_blocks, B, C, N, T = d.shape
        x_hat = d.reshape(B, -1, N, T)  # [B, num_blocks*C, N, T]
        x_hat = self.outfit_module(x_hat)
       
        return x_hat

