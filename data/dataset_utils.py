# -*- coding: utf-8 -*-
"""
数据集工具函数，包含所有数据集通用的数据处理函数
"""
import os
import numpy as np
import pandas as pd
import pickle
import torch
from math import radians, cos, sin, asin, sqrt
from sklearn.preprocessing import StandardScaler
from .dataset_registry import get_dataset_config

def get_norm_params(dataset_name, attr_name):
    """
    获取指定数据集和属性的归一化参数

    Args:
        dataset_name: 数据集名称
        attr_name: 属性名称

    Returns:
        dict: 包含mean, scale, var的归一化参数字典
    """
    config = get_dataset_config(dataset_name)
    if 'norm_params' not in config:
        return None

    # 如果找不到精确匹配，尝试模糊匹配（例如traffic_flow可能匹配traffic_*）
    if attr_name in config['norm_params']:
        return config['norm_params'][attr_name]

    # 对于METR和PEMS数据集，可能只有一个特征，直接返回第一个
    if len(config['norm_params']) == 1:
        return next(iter(config['norm_params'].values()))

    # 尝试模糊匹配
    for key in config['norm_params']:
        if key.lower() in attr_name.lower() or attr_name.lower() in key.lower():
            return config['norm_params'][key]

    return None

def haversine(lon1, lat1, lon2, lat2):
    """
    计算两个经纬度点之间的距离（公里）
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球半径，单位公里
    return c * r


def load_loc_distance(path):
    """
    从CSV/TXT文件加载位置信息，计算距离矩阵
    """
    if path.endswith('.csv'):
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, sep='\t')

    n = len(df)
    A = np.zeros((n, n))

    # 计算站点间的距离
    for i in range(n):
        for j in range(i + 1, n):
            dist = haversine(df.iloc[i]['longitude'], df.iloc[i]['latitude'],
                             df.iloc[j]['longitude'], df.iloc[j]['latitude'])
            A[i, j] = A[j, i] = 1.0 / (dist + 1e-6)  # 距离倒数作为邻接矩阵权重

    # 对角线设为0
    np.fill_diagonal(A, 0)
    return A


def load_loc_direct(path):
    """
    直接从文件加载邻接矩阵
    """
    return np.loadtxt(path)


def load_adj_direct(path):
    """
    直接从npy文件加载邻接矩阵
    """
    return np.load(path)


def load_adj_pickle(path):
    """
    从pickle文件加载邻接矩阵
    """
    with open(path, "rb") as f:
        adj_matrix = pickle.load(f)
    return adj_matrix


def get_node_division(test_nodes_path, num_nodes):
    """
    获取测试节点索引
    """
    if test_nodes_path is None or not os.path.exists(test_nodes_path):
        # 默认使用20%的节点作为测试节点
        num_test = int(0.2 * num_nodes)
        return np.random.choice(num_nodes, num_test, replace=False)
    else:
        return np.load(test_nodes_path)


def load_feature_csv(data_path, dataset_name, time_range, pred_attr):
    """
    从CSV文件加载特征数据，使用注册表中的归一化参数，并确保每个站点的时间不重复
    """
    # 读取数据
    df = pd.read_csv(data_path)

    # 确保时间列为日期时间格式
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])

    # 获取归一化参数
    norm_info = get_norm_params(dataset_name, pred_attr)
    if norm_info is None:
        # 如果没有找到预先记录的归一化参数，就计算
        pred_cols = [col for col in df.columns if col.startswith(pred_attr) and not col.endswith('_Missing')]
        target_data = df[pred_cols].values
        mean = np.mean(target_data)
        scale = np.std(target_data)
        norm_info = {
            'mean': mean,
            'scale': scale,
            'var': scale ** 2
        }

    # 初始化站点数据字典
    station_data = {
        'pred': [],
        'feat': [],
        'missing': [],
        'time': [],
    }

    # 按station_id分组
    grouped_df = df.groupby('station_id')

    # 处理每个站点的数据
    for station_id, station_df in grouped_df:
        # 确保时间不重复 - 对每个站点保留时间唯一的记录
        if 'time' in station_df.columns:
            # 对于重复的时间，保留第一条记录
            station_df = station_df.drop_duplicates(subset=['time'])
            # 按时间排序
            station_df = station_df.sort_values('time')

        # 应用时间范围过滤（如果指定）
        if time_range:
            start_idx = int(len(station_df) * time_range[0])
            end_idx = int(len(station_df) * time_range[1])
            station_df = station_df.iloc[start_idx:end_idx]

        # 提取数据
        raw_data = extract_data_from_df(station_df, pred_attr, norm_info)

        # 添加到站点数据中
        station_data['pred'].append(raw_data['pred'])
        station_data['feat'].append(raw_data['feat'])
        station_data['missing'].append(raw_data['missing'])

        # 只添加一次时间数据
        if not station_data['time'] and raw_data['time'] is not None:
            station_data['time'].append(raw_data['time'])

    # 转换为张量
    station_data['pred'] = torch.FloatTensor(np.array(station_data['pred']))
    station_data['feat'] = torch.FloatTensor(np.array(station_data['feat']))
    station_data['missing'] = torch.FloatTensor(np.array(station_data['missing']))

    return station_data, norm_info

def extract_data_from_df(df, pred_attr, norm_params=None):
    """
    从DataFrame中提取数据，不进行填充
    """
    # 提取预测目标、特征和缺失掩码
    pred_cols = [col for col in df.columns if col.startswith(pred_attr) and not col.endswith('_Missing')]
    feat_cols = [col for col in df.columns if not col in pred_cols
                 and not col.endswith('_Concentration') and not col.endswith('_Missing')
                 and col not in ['time', 'station_id']]
    missing_cols = [col for col in df.columns if col.startswith(pred_attr) and col.endswith('_Missing')]

    # 提取数据
    target_data = df[pred_cols].values

    # 归一化目标数据
    mean = norm_params['mean']
    scale = norm_params['scale']
    target_data_norm = (target_data - mean) / scale

    # 构建特征数据
    features = df[feat_cols].values if feat_cols else None

    # 构建缺失掩码
    missing = df[missing_cols].values if missing_cols else np.ones_like(target_data)

    # 时间数据
    time_data = df['time'].values if 'time' in df.columns else None

    # 构建原始数据字典
    raw_data = {
        'pred': target_data_norm,
        'feat': features,
        'missing': missing,
        'time': time_data
    }

    return raw_data

def load_feature_npz(data_path, dataset_name, time_range, pred_attr):
    """
    从NPZ文件加载特征数据，使用注册表中的归一化参数
    """
    # 加载数据
    data = np.load(data_path)
    x = data['x']  # 假设数据格式为 (num_nodes, num_timesteps, num_features)

    # 尝试加载时间信息（如果存在）
    time_data = data.get('time', None)

    # 应用时间范围过滤
    if time_range:
        start_idx = int(x.shape[1] * time_range[0])
        end_idx = int(x.shape[1] * time_range[1])
        x = x[:, start_idx:end_idx, :]
        if time_data is not None:
            time_data = time_data[start_idx:end_idx]

    # 获取归一化参数
    norm_params = get_norm_params(dataset_name, pred_attr)

    # 提取第一个特征作为预测目标
    target_data = x[:, :, :1]
    features = x[:, :, 1:] if x.shape[-1] > 1 else None

    # 尝试加载缺失值掩码（如果存在）
    missing_mask = data.get('missing_mask', None)
    if missing_mask is None:
        # 如果没有提供缺失值掩码，默认为全部有效（1）
        missing_mask = np.ones_like(target_data)
    elif missing_mask.shape != target_data.shape:
        # 如果形状不匹配，可能需要调整
        if len(missing_mask.shape) == 2:  # 如果是二维的
            missing_mask = missing_mask.reshape(*missing_mask.shape, 1)
        # 应用时间范围过滤到缺失掩码
        if time_range:
            missing_mask = missing_mask[:, start_idx:end_idx, :]

    # 归一化目标数据
    if norm_params:
        # 使用预定义的归一化参数
        mean = norm_params['mean']
        scale = norm_params['scale']
    else:
        # 如果没有预定义参数，从数据中计算
        mean = np.mean(target_data)
        scale = np.std(target_data)
        if scale < 1e-10:
            scale = 1.0  # 避免除以接近零的值

    # 归一化
    target_data_norm = (target_data - mean) / scale

    # 构建原始数据字典，保持与load_feature_csv一致的格式
    station_data = {
        'pred': [target_data_norm],  # 列表中包含一个数组
        'feat': [features] if features is not None else [],
        'missing': [missing_mask],
        'time': [time_data] if time_data is not None else [],
    }

    # 转换为PyTorch张量
    station_data['pred'] = torch.FloatTensor(np.array(station_data['pred']))
    if station_data['feat']:
        station_data['feat'] = torch.FloatTensor(np.array(station_data['feat']))
    else:
        station_data['feat'] = torch.FloatTensor([])
    station_data['missing'] = torch.FloatTensor(np.array(station_data['missing']))

    # 返回数据和归一化信息
    norm_info = {
        'mean': mean,
        'scale': scale,
        'var': scale ** 2
    }

    return station_data, norm_info
