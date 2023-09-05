# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 16:29:48 2023

@author: LUN076
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class GAT(nn.Module):
    def __init__(self, n_feat, n_hid, n_heads=4, dropout=0.01, alpha=0.01):
        """Dense version of GAT
        """
        super(GAT, self).__init__()
        n_out = n_feat
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(n_feat, n_hid, dropout=dropout, alpha=alpha, concat=True) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(n_hid * n_heads, n_out, dropout=dropout,alpha=alpha, concat=False)
    
    def forward(self, x, adj):
        # x: [b, n, t, c]
        b, n, t, c = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b*t, n, c)  # [b*t, n, c]
        x = F.dropout(x, self.dropout, training=self.training)  
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)  # 将每个head得到的表示进行拼接
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))   # 输出并激活
        x = x.reshape(b, n, t, c) #恢复形状
        return x

class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()

class GraphAttentionLayer(nn.Module):
    """
    图注意力层
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features   # 节点表示向量的输入特征数
        self.out_features = out_features   # 节点表示向量的输出特征数
        self.dropout = dropout    # dropout参数
        self.alpha = alpha     # leakyrelu激活的参数
        self.concat = concat   # 如果为true, 再进行elu激活
        
        # 定义可训练参数，即论文中的W和a
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))  
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # 初始化
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)   # 初始化
        # 定义leakyrelu激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(self, inp, adj):
        """
        inp: input_fea [B, N, in_features]  in_features表示节点的输入特征向量元素个数
        adj: 图的邻接矩阵  [N, N] 非零即一，数据结构基本知识
        """
        h = torch.matmul(inp, self.W)   # [B, N, out_features] 384 10 16
        N = h.size()[1]    # N 图的节点数
        a_input = torch.cat([h.repeat(1,1,N).view(-1, N*N, self.out_features), h.repeat(1, N, 1)], dim=-1)
        a_input = a_input.view(-1, N, N, 2*self.out_features)
        # [B, N, N, 2*out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        # [B, N, N, 1] => [B, N, N] 图注意力的相关系数（未归一化）        
        zero_vec = -1e12 * torch.ones_like(e)    # 将没有连接的边置为负无穷
        attention = torch.where(adj>0, e, zero_vec)   # [B, N, N]
        # 表示如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留，
        # 否则需要mask并置为非常小的值，原因是softmax的时候这个最小值会不考虑。
        attention = F.softmax(attention, dim=1)    # softmax形状保持不变 [B, N, N]，得到归一化的注意力权重！
        attention = F.dropout(attention, self.dropout, training=self.training)   # dropout，防止过拟合
        h_prime = torch.matmul(attention, h)  # [B, N, N].[B, N, out_features] => [B, N, out_features]
        # 得到由周围节点通过注意力权重进行更新的表示
        if self.concat:
            return F.relu(h_prime)
        else:
            return h_prime 
    
if __name__ == "__main__":

    import torch.nn as nn
    import torch.optim as optim
    import numpy as np

    from torch.utils.data import DataLoader

    # 定义超参数
    batch_size = 16
    learning_rate = 1e-4
    num_epochs = 100

    num_samples = 100
    data_shape = (10, 24, 16)
    dummy_data = torch.ones(num_samples, *data_shape)
    adj = torch.tensor(np.random.rand(10, 10))

    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(dummy_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 创建模型
    model = GAT(n_feat=16, n_hid=32, n_heads=4)

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
            outputs = model(inputs, adj)
            loss = criterion(outputs, inputs)  # 使用均方误差计算损失
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        average_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {average_loss:.4f}")

    # 保存模型
    # torch.save(model.state_dict(), 'ds_msa_model.pth')
 
    
    