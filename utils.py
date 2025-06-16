# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 15:22:04 2023

@author: dell
"""
import torch
import numpy as np
#####################################
# evaluation metrics
#####################################
import torch
import matplotlib.pyplot as plt

def get_info(opt, y_pred, label, mask, mask_impute=None):


    print("scale: ", opt.scale)
    print("mean: ", opt.mean)
    print("var: ", opt.var)
    print(isinstance(y_pred, torch.Tensor))

    if isinstance(y_pred, list): y_pred = torch.cat(y_pred, dim=0)
    if isinstance(label, list): label = torch.cat(label, dim=0)
    if isinstance(mask, list): mask = torch.cat(mask, dim=0)
    if mask_impute is not None and isinstance(mask_impute, list): mask_impute = torch.cat(mask_impute, dim=0)

    if not isinstance(y_pred, torch.Tensor): y_pred = torch.tensor(y_pred)
    if not isinstance(label, torch.Tensor): label = torch.tensor(label)
    if not isinstance(mask, torch.Tensor): mask = torch.tensor(mask)
    if mask_impute is not None and not isinstance(mask_impute, torch.Tensor): mask_impute = torch.tensor(mask_impute)


    y = y_pred * opt.scale + opt.mean
    label =label * opt.scale  + opt.mean
    # for impute

    if len(y.shape) == 4: y = y.squeeze(2) if y.shape[2] == 1 else y.squeeze(-1)
    if len(label.shape) ==4: label = label.squeeze(2) if label.shape[2] == 1 else label.squeeze(-1)
    if len(mask.shape) ==4: mask = mask.squeeze(2) if mask.shape[2] == 1 else mask.squeeze(-1)
    if mask_impute is not None and len(mask_impute.shape) == 4: mask_impute = mask_impute.squeeze(2) if mask_impute.shape[2] == 1 else mask_impute.squeeze(-1)

    # mask 1 为缺失
    # mask_impute 0 为缺失
    val = (1-mask)*(1-mask_impute) if mask_impute is not None else 1-mask
    print("mae: ", _mae_with_missing(y, label, val))
    print("rmse: ", _rmse_with_missing(y, label, val))
    print("mape: ", _mape_with_missing(y, label, val))
    plot_groups(y, label, val)

def plot_groups(data, labels, val, group_num=3):
    """
    data: tuple of (y_pred, label) tensors with shape [b, n, t, 1]
    labels: tuple of legend labels ('Pred', 'True')
    """
    # 合并batch和时间维度 [b, n,, t] -> [n, b*t]
    b, n, t = data.shape
    data = data.permute(1, 0, 2).reshape(n, -1).contiguous().cpu()
    labels = labels.permute(1, 0, 2).reshape(n, -1).contiguous().cpu()
    val = val.permute(1, 0, 2).reshape(n, -1).contiguous().cpu()

#
    data = ((data)*(val) + labels*(1-val)).numpy()[:, :200]

    labels = labels.numpy()[:, :200]



    for i in range(group_num):
        plt.plot([i for i in range(len(labels[i]))], labels[i], 'r-', linewidth=0.2, label='True Label')
        plt.plot([i for i in range(len(labels[i]))], data[i], 'o-', linewidth=0.2, label='Pred',
                 markersize=0.2)
        plt.legend()
        plt.show()



def _rmse_with_missing(y, label, missing_mask):
    """
    Args:
        y: Tensor [time, num_m, dy]
        label: Tensor
        missing_mask: [time, num_m, 1] or [time, num_m]
    Returns:
        rmse: float scalar tensor
    """
    # 维度对齐处理
    y = y.cpu()
    label = label.cpu()
    missing_mask = missing_mask.cpu()

    squared_error = (y - label).pow(2) * missing_mask
    valid_count = missing_mask.sum().clamp(min=1e-7)

    rmse = torch.sqrt(squared_error.sum() / valid_count)
    return rmse


def _mae_with_missing(y, label, missing_mask):
    """
    Args:
        y: Tensor [time, num_m, dy]
        label: Tensor
        missing_mask: [time, num_m, 1] or [time, num_m]
    Returns:
        mae: float scalar tensor
    """
    # 自动广播mask维度
    # 维度对齐处理


    y = y.cpu()
    label = label.cpu()
    missing_mask = missing_mask.cpu()

    abs_error = torch.abs(y - label) *missing_mask
    valid_count =  missing_mask.sum().clamp(min=1e-7)

    mae = abs_error.sum() / valid_count
    return mae


def _mape_with_missing(y, label, missing_mask):
    """
    Args:
        y: Tensor [time, num_m, dy]
        label: Tensor
        missing_mask: [time, num_m, 1] or [time, num_m]
        eps: 防止除零的小量
    Returns:
        mape: float scalar tensor
    """

    y = y.cpu()
    label = label.cpu()
    missing_mask = missing_mask.cpu()

    relative_error = ((torch.abs((y - label)/label))*missing_mask).sum() /  missing_mask.sum()
    return relative_error


def _quantile_CRPS_with_missing(y, label, missing_mask):


    y = torch.concat(y, dim=0) # b nsample n t
    label = torch.concat(label, dim=0).squeeze(2) # b n c t
    missing_mask = torch.concat(missing_mask, dim=0).squeeze(2)

    b, nsample, n, t = y.shape


    y = y.permute(0, 3, 2, 1) # b t n nsample
    label = label.permute(0, 2, 1) # b t n
    missing_mask = missing_mask.permute(0, 2, 1)

    y = y.reshape(b*t, n, nsample).permute(2, 0, 1)
    label = label.reshape(b*t, n)
    missing_mask = missing_mask.reshape(b*t, n)
    """
    Args:
        y: Tensor [num_sample, time, num_m, dy]
        label: Tensor [time, num_m, dy]
        missing_mask: [time, num_m, 1] or [time, num_m]
    Returns:
        CRPS: float scalar tensor
    """

    def quantile_loss(target, forecast, q, eval_points):
        indicator = (target <= forecast).float()
        return 2 * torch.abs((forecast - target) * eval_points * (indicator - q)).sum()

    def calc_denominator(target, valid_mask):
        return torch.abs(target * valid_mask).sum().clamp(min=1e-7)

    # 维度对齐
    if missing_mask.dim() != label.dim():
        missing_mask = missing_mask.unsqueeze(-1)
    valid_mask = missing_mask.float()

    # 生成分位数点 (保持设备一致)
    quantiles = torch.linspace(0.05, 0.95, 19, device=y.device)
    denominator = calc_denominator(label, valid_mask)

    crps = torch.tensor(0.0, device=y.device)
    for q in quantiles:
        q_pred = torch.quantile(y, q, dim=0)
        q_loss = quantile_loss(label, q_pred, q, valid_mask)
        crps += q_loss / denominator

    return crps / len(quantiles)


def generate_missing_mask(shape,
                          missing_ratio=0.3,
                          continuous=False,
                          block_size=3,
                          time_dim=-1,
                          device='cpu'):
    """
    生成可调节缺失模式的mask矩阵
    Args:
        shape (tuple): 目标张量形状，通常为(batch, nodes, features, timesteps)
        missing_ratio (float): 总缺失比例，默认0.3
        continuous (bool): 是否启用连续缺失模式，默认False
        block_size (int): 连续缺失块长度，仅在continuous=True时有效
        time_dim (int): 时间维度位置，默认-1（最后一个维度）
        device (str): 输出设备，默认'cpu'
    Returns:
        mask (Tensor): 与输入shape相同的二进制mask，1表示观测存在，0表示缺失
    """
    # 初始化全1 mask
    mask = torch.ones(shape, device=device)
    total_elements = mask.numel()
    num_missing = int(total_elements * missing_ratio)

    if num_missing == 0:
        return mask

    if not continuous:
        # 完全随机缺失模式
        flat_mask = mask.view(-1)
        indices = torch.randperm(flat_mask.size(0), device=device)[:num_missing]
        flat_mask[indices] = 0
        mask = flat_mask.view(shape)
    else:
        # 连续块缺失模式
        time_steps = shape[time_dim]
        num_blocks = max(1, num_missing // block_size)

        # 生成随机块起始位置
        batch_size, num_nodes, num_feat, _ = shape
        batch_idx = torch.randint(0, batch_size, (num_blocks,), device=device)
        node_idx = torch.randint(0, num_nodes, (num_blocks,), device=device)
        feat_idx = torch.randint(0, num_feat, (num_blocks,), device=device)
        time_idx = torch.randint(0, time_steps - block_size + 1, (num_blocks,), device=device)

        # 应用块缺失
        for i in range(num_blocks):
            if time_dim == -1:
                mask[batch_idx[i], node_idx[i], feat_idx[i],
                time_idx[i]:time_idx[i] + block_size] = 0
            else:
                # 处理不同时间维度位置
                slice_idx = [slice(None)] * mask.ndim
                slice_idx[time_dim] = slice(time_idx[i], time_idx[i] + block_size)
                mask[tuple(slice_idx)] = 0

        # 补充剩余缺失点
        current_missing = (mask == 0).sum().item()
        if current_missing < num_missing:
            additional_missing = num_missing - current_missing
            flat_mask = mask.view(-1)
            indices = torch.where(flat_mask == 1)[0]
            rand_idx = torch.randperm(indices.size(0), device=device)[:additional_missing]
            flat_mask[indices[rand_idx]] = 0
            mask = flat_mask.view(shape)

    return mask