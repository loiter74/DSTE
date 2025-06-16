# -*- coding: utf-8 -*-
"""
数据集加载和可视化工具
用于检查数据集的结构和内容
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import namedtuple
import argparse

from draw import plot_line, plot_pm25_heatmap

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入数据集相关模块
from data.dataset_factory import GenericDataset
from data.dataset_registry import get_dataset_config, DATASET_CONFIG

# 创建一个简单的选项对象来模拟命令行参数
Options = namedtuple('Options', ['dataset', 'pred_attr', 'batch_size'])


def visualize_dataset(opt):
    """
    加载并可视化数据集

    Args:
        dataset_name: 数据集名称
        pred_attr: 预测属性名称，如果为None则使用第一个属性
    """
    # 获取数据集配置
    config = get_dataset_config(opt.dataset_name)

    # 如果未指定预测属性，使用第一个属性
    if opt.pred_attr is None:
        if 'all_attrs' in config and config['all_attrs']:
            opt.pred_attr = config['all_attrs'][0]
        else:
            opt.pred_attr = "traffic_flow"  # 默认值

    print(f"加载数据集: {opt.dataset_name}")
    print(f"预测属性: {opt.pred_attr}")
    # 加载数据集
    try:
        dataset = GenericDataset(opt, mode='train')
        print("\n数据集加载成功!")

        # 显示数据集基本信息
        print("\n=== 数据集基本信息 ===")
        print(f"节点数量: {dataset.raw_data['pred'].shape[0]}")
        print(f"时间步数量: {dataset.raw_data['pred'].shape[1]}")
        print(f"预测目标维度: {dataset.raw_data['pred'].shape[2]}")

        if dataset.raw_data['feat'] is not None:
            print(f"特征维度: {dataset.raw_data['feat'].shape[2]}")
        else:
            print("特征: 无")

        print(f"训练节点数量: {len(dataset.train_node_index)}")
        print(f"测试节点数量: {len(dataset.test_node_index)}")
        print(f"测试节点索引: {dataset.test_node_index}")

        # 显示归一化信息
        print("\n=== 归一化参数 ===")
        print(f"均值: {dataset.norm_info['mean']}")
        print(f"标准差: {dataset.norm_info['scale']}")
        print(f"方差: {dataset.var if hasattr(dataset, 'var') else dataset.norm_info['scale'] ** 2}")

        # 显示邻接矩阵信息
        print("\n=== 邻接矩阵信息 ===")
        print(f"形状: {dataset.A.shape}")
        print(f"非零元素数量: {np.count_nonzero(dataset.A)}")
        print(f"密度: {np.count_nonzero(dataset.A) / (dataset.A.shape[0] * dataset.A.shape[1]):.4f}")

        plot_line(dataset.raw_data['time'][0], dataset.raw_data['pred'][0][:].squeeze(-1), title=None, xlabel=None, ylabel=None, save_path=None)
        # 使用示例:
        # 假设 dataset.raw_data['time'][0] 是时间数组
        # dataset.raw_data['pred'][0] 形状为 [35, num_timesteps, 1]
        timestamps = dataset.raw_data['time'][0]
        plot_pm25_heatmap(
            time_data=pd.to_datetime(timestamps, unit='s'),
            pred_data=dataset.raw_data['pred'],
            title="各站点PM2.5浓度随时间变化热力图",
            xlabel="时间",
            ylabel="站点ID",
            save_path="pm25_heatmap.png"
        )
    except Exception as e:
        print(e)


def main():
    parser = argparse.ArgumentParser(description='数据集加载和可视化工具')
    parser.add_argument('--dataset_name', type=str, default="BJAir2017",
                        help='数据集名称')
    parser.add_argument('--pred_attr', type=str, default="PM25",
                        help='预测属性名称')
    parser.add_argument('--t_len', type=int, default=24, help='time window for inference')
    args = parser.parse_args()

    # 显示可用的数据集
    print("可用的数据集:")
    for name in DATASET_CONFIG.keys():
        print(f"- {name}")

    # 加载并可视化指定的数据集
    visualize_dataset(args)


if __name__ == "__main__":
    main()
