import argparse
from cProfile import label
from multiprocessing import freeze_support

import numpy as np
import torch
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsRegressor
from data.dataset_BJAir2017 import get_dataloader
from data.dataset_PEMS03_small import get_dataloader
from utils import _mae_with_missing, _rmse_with_missing, _mape_with_missing, get_info, generate_missing_mask


def calculate_metrics(original_data, filled_data, mask):
    """
    original_data: 原始数据 (numpy array)
    filled_data: 插补后的数据 (numpy array)
    mask: 掩码矩阵 (numpy array)，1 表示观测值，0 表示缺失值
    """
    # 找到缺失值的位置（mask == 0 的位置是插补值）
    missing_mask = (mask == 0)

    # 计算 MAE
    mae = np.abs(original_data[missing_mask] - filled_data[missing_mask]).mean()

    # 计算 RMSE
    rmse = np.sqrt(np.mean((original_data[missing_mask] - filled_data[missing_mask]) ** 2))

    return mae, rmse

if __name__ == '__main__':
    freeze_support()
    parser = argparse.ArgumentParser(description="DSTE")
    parser.add_argument('--pred_attr', type=str, default="PM10_Concentration", help='Which AQ attribute to infer')
    parser.add_argument('--t_len', type=int, default=24, help='time window for inference')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--num_train_target', type=int, default=3, help='# of target set for training at the start')

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42, help='random seed for initialization')
    # parser.add_argument('--missing_rate', type=float, default=0.6)

    opt = parser.parse_args()

    train_loader, valid_loader, test_loader = get_dataloader(opt)


    # 初始化存储原始数据、插补数据和掩码的列表
    all_original_data = []
    all_filled_data = []
    all_masks = []
    all_val = []

    missing_ratio = 0.5
    knn = KNNImputer(n_neighbors=20, weights="uniform")  # KNN 插补器，使用 3 个邻居
    for data in valid_loader:  # inner loop within one epoch
        pred_target = data["pred_target"]
        mask0 = generate_missing_mask(pred_target.shape, missing_ratio=missing_ratio).numpy()
        b, n, c, t = pred_target.shape

        data_missing = (pred_target*mask0).numpy()
        data_missing[mask0 == 0] = np.nan
        data_reshaped = data_missing.reshape(-1, t) # 切换 n/ t 使用时间/空间维度
        data_filled = knn.fit_transform(data_reshaped).reshape(b, n, c, t)

        # 将当前批次的数据添加到列表中
        all_original_data.append(pred_target.numpy())
        all_filled_data.append(data_filled)
        all_masks.append(mask0)
        all_val.append(data["missing_mask_target"])

    # 将所有批次的数据拼接成单一数组
    all_original_data = np.concatenate(all_original_data, axis=0)  # 拼接原始数据
    all_filled_data = np.concatenate(all_filled_data, axis=0)    # 拼接插补数据
    all_masks = np.concatenate(all_masks, axis=0)                 # 拼接掩码
    all_val_mask = torch.concat(all_val, dim=0)

    get_info(opt, all_filled_data, all_original_data, all_val_mask, all_masks)
    print(missing_ratio)