# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 12:11:24 2023

@author: dell
"""
import torch
import torch.nn as nn
from torch.nn import Conv1d

from model.inner.st_module.conv_layer import Conv1d_with_init
from model.inner.st_module.graph_layer import DynamicAgg, StaticAgg


class ImLinearBase(nn.Module):
    def __init__(self, config):
        super().__init__()

        # 特征嵌入层
        self.input_proj = Conv1d_with_init(config['input_dim'], config['hidden_channel'])
        self.output_proj = nn.Sequential(Conv1d(config['hidden_channel'], config['hidden_channel']//2, kernel_size=1),
                                         Conv1d(config['hidden_channel']//2, config['hidden_channel'] // 4, kernel_size=1),
                                         Conv1d(config['hidden_channel'] // 4, config['input_dim'], kernel_size=1),
                                         )

        # 双向lstm
        # batch_size, sequence_length, features
        self.linear_model = getLSTM(2*config['hidden_channel'], config['hidden_channel']//2, bidirectional=True)

        # TCN
        #self.linear_model = MultiTConv1d(2*config['hidden_channel'], config['hidden_channel'], config['hidden_channel'], num_layers=6)

        # attn
        #self.linear_model = TemporalLearning()
        #
        self.graph_aggeration = DynamicAgg(pred_in=1, feat_in=7, channels=config['hidden_channel'], out=config['hidden_channel'])
        self.graph_aggeration2 = StaticAgg(pred_in=1, channels=config['hidden_channel'], out_channels=config['hidden_channel'])

    def forward(self, x_context, y_context, x_target, y_target, adj, missing_mask, impute_mask):
        # 输入预处理
        b, n, c, t = y_target.shape


        y = self.input_proj(y_target)

        graph_agg = self.graph_aggeration(x_target, x_context, y_context)
        graph_agg2 = self.graph_aggeration2(y_context, adj[:, 1])
        y = torch.cat((y, y), dim=2)

        y = y.flatten(0, 1)

        y = y.permute(0, 2, 1) # b*n, c ,t
        y, _ = self.linear_model(y)
        y = y.permute(0, 2, 1)

       # y = self.linear_model(y)
        y = self.output_proj(y).reshape(b, n, c, t)

        return y

#mae:  tensor(11.3390) 有grap 100ep
# rmse:  tensor(18.9972)
# mape:  tensor(0.3127)
    def compute_loss(self, pred, target, mask):

        valid_index = mask.permute(0, 2, 1).unsqueeze(2)
        mse_loss = ((pred - target) ** 2 * valid_index).sum() / valid_index.sum()

        # 对observation模块的所有参数添加L2正则
        reg_loss = 0.0
        for param in self.parameters():
            reg_loss += torch.norm(param, p=2)  # L2正则

        return mse_loss , mse_loss, reg_loss  # 调节0.01为合适系数+ 0.001 * reg_loss

    def compute_loss_impute(self, pred, target, mask, mask0):


        valid_index = (1 - mask.permute(0, 2, 1).unsqueeze(2))*(1-mask0)
        mse_loss = ((pred - target) ** 2 * valid_index).sum() / valid_index.sum()

        # 对observation模块的所有参数添加L2正则
        reg_loss = 0.0
        for param in self.parameters():
            reg_loss += torch.norm(param, p=2)  # L2正则

        return mse_loss, mse_loss, reg_loss  # 调节0.01为合适系数+ 0.001 * reg_loss

def getLSTM(input_size, hidden_size, bidirectional):
    return nn.LSTM(input_size= input_size,
                                    hidden_size=hidden_size,
                                    num_layers=4,
                                    batch_first=True,
                                    dropout=0.1,
                                    bidirectional=bidirectional)