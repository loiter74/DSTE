# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 16:33:10 2023

@author: dell
"""
import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Embedding, self).__init__()
        # 定义三个全连接层，逐渐升维到 32 维
        self.fc1 = nn.Linear(input_dim, 128)  # 输入维度为 input_dim，输出维度为 128
        self.fc2 = nn.Linear(128, 256)  # 输入维度为 128，输出维度为 256
        self.fc3 = nn.Linear(256, output_dim)  # 输入维度为 256，输出维度为 output_dim

    def forward(self, x):
        # 前向传播
        x = torch.relu(self.fc1(x))  # 第一个全连接层，使用 ReLU 激活函数
        x = torch.relu(self.fc2(x))  # 第二个全连接层，使用 ReLU 激活函数
        x = self.fc3(x)  # 第三个全连接层，没有激活函数
        return x


if __name__ == "__main__":
    # 创建模型实例
    input_dim = 3  # 输入维度为 3
    output_dim = 32  # 输出维度为 32，与目标维度匹配
    model = Embedding(input_dim, output_dim)
    
    # 创建输入张量，假设 (b, n, t) 为样本数量、特征数量和时间步数
    b, n, t = 32, 64, 10  # 示例输入张量的形状为 (b, n, t, 3)
    input_tensor = torch.randn(b, n, t, input_dim)
    
    # 将输入张量通过模型前向传播
    output_tensor = model(input_tensor.view(b * n * t, input_dim)).view(b, n, t, output_dim)
    
    # 输出张量的形状为 (b, n, t, output_dim)，即 (32, 64, 10, 32)
    print("输出张量的形状：", output_tensor.shape)
