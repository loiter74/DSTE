"""
This module implements an abstract base class (ABC) 'BaseDataset' for datasets.
It also includes common transformation functions, which can be later used in subclasses.
"""
import random
import torch
import torch.utils.data as data
from abc import ABC, abstractmethod
import numpy as np
import os
from math import radians, cos, sin, asin, sqrt
import scipy.sparse as sp
from scipy.sparse import linalg

class BaseDataset(data.Dataset, ABC):
    def __init__(self, opt, mode='train'):
        self.t_len = opt.t_len
        self.mode = mode
        self.time_division = {
            'train': [0.0, 0.8],
            'val': [0.8, 0.9],
            'test': [0.9, 1.0]
        }
        self.raw_data = {}
        self.A = None
        self.test_node_index = None
        self.train_node_index = None
        self.norm_info = {}  # 存储归一化信息

    def __len__(self):
        """Return the total number of data points."""
        # 计算可用的时间步长
        if 'pred' in self.raw_data:
            return max(0, self.raw_data['pred'].shape[1] - self.t_len)
        return 0

    def __getitem__(self, index):
        # 获取预测目标和特征数据
        pred_data = self.raw_data['pred']
        feat_data = self.raw_data.get('feat', None)
        missing_mask = self.raw_data.get('missing', None)

        # 如果没有缺失掩码，创建一个全1的掩码（表示没有缺失）
        if missing_mask is None:
            missing_mask = np.ones_like(pred_data)

        # 获取当前时间窗口的数据
        pred_window = pred_data[:, index:index+self.t_len]

        if feat_data is not None:
            feat_window = feat_data[:, index:index+self.t_len]
        else:
            feat_window = np.zeros((pred_window.shape[0], self.t_len, 0))

        missing_window = missing_mask[:, index:index+self.t_len]

        # 划分上下文节点和目标节点
        if self.mode == 'train':
            # 训练模式：随机选择目标节点
            target_size = int(len(self.train_node_index) * 0.3)
            target_idx = np.random.choice(len(self.train_node_index), size=target_size, replace=False)
            target_nodes = self.train_node_index[target_idx]
            context_nodes = np.setdiff1d(np.arange(pred_data.shape[0]), target_nodes)
        else:
            # 验证/测试模式：使用测试节点作为目标
            target_nodes = self.test_node_index
            context_nodes = self.train_node_index

        # 提取上下文和目标数据
        pred_context = pred_window[context_nodes]
        feat_context = feat_window[context_nodes]
        missing_mask_context = missing_window[context_nodes]

        pred_target = pred_window[target_nodes]
        feat_target = feat_window[target_nodes]
        missing_mask_target = missing_window[target_nodes]

        # 创建邻接矩阵
        # 创建包含一阶和二阶邻居的多通道邻接矩阵
        order = 2
        adj = self.get_multi_order_adj(self.A.copy(), orders=2)  # [2, n, n]

        # 创建 target 和 context 节点之间的邻接矩阵
        adj_tc = np.zeros((order, len(target_nodes), len(context_nodes)))
        # 对每个通道提取子图
        for i in range(order):
            adj_tc[i] = adj[i][np.ix_(target_nodes, context_nodes)]


        # 转换为张量
        pred_context_tensor = torch.FloatTensor(pred_context)
        feat_context_tensor = torch.FloatTensor(feat_context)
        pred_target_tensor = torch.FloatTensor(pred_target)
        feat_target_tensor = torch.FloatTensor(feat_target)
        adj_tensor = torch.FloatTensor(adj)
        adj_tc = torch.FloatTensor(adj_tc)
        missing_mask_context_tensor = torch.FloatTensor(missing_mask_context)
        missing_mask_target_tensor = torch.FloatTensor(missing_mask_target)

        return {
            'pred_context': pred_context_tensor,
            'feat_context': feat_context_tensor,
            'pred_target': pred_target_tensor,
            'feat_target': feat_target_tensor,
            'adj': adj_tensor,
            'adj_tc': adj_tc,
            'missing_mask_context': missing_mask_context_tensor,
            'missing_mask_target': missing_mask_target_tensor,
            'context_nodes': context_nodes,
            'target_nodes': target_nodes
        }

    def normalize_adj(self, adj):
        """
        对邻接矩阵进行归一化处理
        """
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).toarray()

    def get_multi_order_adj(self, A, orders=2):
        """
        创建包含一阶和二阶邻居的多通道邻接矩阵

        参数:
        - A: 原始邻接矩阵，形状为 [n, n]
        - orders: 需要的阶数，默认为2（一阶和二阶）

        返回:
        - multi_adj: 多通道邻接矩阵，形状为 [orders, n, n]
        """
        n = A.shape[0]
        # 初始化多通道邻接矩阵
        multi_adj = np.zeros((orders, n, n))

        # 复制并归一化原始邻接矩阵作为一阶邻居
        adj_1 = self.normalize_adj(A.copy())
        multi_adj[0] = adj_1

        # 计算二阶邻居（邻接矩阵的平方）并归一化
        if orders >= 2:
            # 使用矩阵乘法计算二阶连接
            adj_2 = np.matmul(adj_1, adj_1)
            # 可选：移除自环（对角线元素）
            np.fill_diagonal(adj_2, 0)
            # 归一化二阶邻接矩阵
            adj_2 = self.normalize_adj(adj_2)
            multi_adj[1] = adj_2

        # 如果需要更高阶的邻居，可以继续添加
        for i in range(2, orders):
            adj_higher = np.matmul(multi_adj[i - 1], adj_1)
            np.fill_diagonal(adj_higher, 0)
            adj_higher = self.normalize_adj(adj_higher)
            multi_adj[i] = adj_higher

        return multi_adj


