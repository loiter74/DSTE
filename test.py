# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 16:40:46 2023

@author: LUN076
"""

import unittest
import torch

# 导入你的模块
from Spatial_Layer import SpatialAttention, DS_MSA

class TestSpatialAttention(unittest.TestCase):

    def test_spatial_attention(self):
        # 创建一个SpatialAttention层的实例
        dim = 64
        num_heads = 4
        num_sectors = 17
        assignment = torch.randn(1, num_sectors, num_sectors)
        mask = torch.randn(1, num_sectors)
        attn_layer = SpatialAttention(dim, heads=num_heads, assignment=assignment, mask=mask)

        # 创建一个随机输入张量
        batch_size = 2
        sequence_length = 10
        input_dim = dim
        x = torch.randn(batch_size, sequence_length, input_dim)

        # 检查前向传播是否正常工作
        output = attn_layer(x)
        self.assertEqual(output.shape, (batch_size, sequence_length, input_dim))

    def test_ds_msa(self):
        # 创建一个DS_MSA层的实例
        dim = 64
        depth = 3
        num_heads = 4
        mlp_dim = 128
        assignment = torch.randn(1, 17, 17)
        mask = torch.randn(1, 17)
        ds_msa_layer = DS_MSA(dim, depth, num_heads, mlp_dim, assignment, mask)

        # 创建一个随机输入张量
        batch_size = 2
        sequence_length = 10
        input_dim = dim
        time_steps = 5
        x = torch.randn(batch_size, input_dim, sequence_length, time_steps)

        # 检查前向传播是否正常工作
        output = ds_msa_layer(x)
        self.assertEqual(output.shape, (batch_size, input_dim, sequence_length, time_steps))

if __name__ == '__main__':
    unittest.main()
