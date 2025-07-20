# -*- coding: utf-8 -*-
"""
数据集工厂，根据配置创建不同的数据集实例
"""

import torch
from torch.utils.data import DataLoader
from .dataset_base import BaseDataset
from .dataset_registry import get_dataset_config
from .dataset_utils import *

class GenericDataset(BaseDataset):
    """
    通用数据集类，替代原来的多个特定数据集类
    """

    def __init__(self, opt, mode='train'):
        super().__init__(opt, mode)

        # 获取数据集配置
        config = get_dataset_config(opt.dataset_name)

        # 设置预测属性和需要排除的属性
        self.pred_attrs = opt.pred_attr

        if 'all_attrs' in config and config['all_attrs']:
            self.drop_attrs = config['all_attrs'].copy()
            self.drop_attrs = [attr for attr in self.drop_attrs if not attr.startswith(opt.pred_attr)]
            if 'missing_attrs' in config and config['missing_attrs']:
                self.drop_attrs += config['missing_attrs']

        # 加载邻接矩阵
        loc_method = config.get('load_loc_method')
        if loc_method == 'load_loc_distance':
            self.A = load_loc_distance(config['location_path'])
        elif loc_method == 'load_loc_direct':
            self.A = load_loc_direct(config['location_path'])
        elif loc_method == 'load_adj_direct':
            self.A = load_adj_direct(config['adj_path'])
        elif loc_method == 'load_adj_pickle':
            self.A = load_adj_pickle(config['adj_path'])

        # 加载特征数据与目标特征归一化信息
        if config['data_path'].endswith('.csv') or config['data_path'].endswith('.txt'):
            self.raw_data, self.norm_info = load_feature_csv(
                config['data_path'],
                opt.dataset_name,
                self.time_division[mode],
                self.pred_attrs
            )
        elif config['data_path'].endswith('.npz') or config['data_path'].endswith('.npy'):
            self.raw_data, self.norm_info = load_feature_npz(
                config['data_path'],
                opt.dataset_name,
                self.time_division[mode],
                self.pred_attrs
            )

        # 获取训练/测试节点划分
        self.test_node_index = get_node_division(
            config.get('test_nodes_path'),
            num_nodes=len(self.raw_data['pred'])
        )
        self.train_node_index = np.setdiff1d(
            np.arange(len(self.raw_data['pred'])),
            self.test_node_index
        )


def get_dataloader(opt):
    """
    获取数据加载器
    """
    train_dataset = GenericDataset(opt, mode='train')
    val_dataset = GenericDataset(opt, mode='val')
    test_dataset = GenericDataset(opt, mode='test')

    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=4
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=4
    )

    scale = val_dataset.norm_info['scale']
    mean = val_dataset.norm_info['mean']
    var = val_dataset.norm_info['var'] if 'var' in val_dataset.norm_info else None
    return train_loader, val_loader, test_loader, scale, mean, var
