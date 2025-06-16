# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 13:55:00 2023

@author: LUN076
"""
# from data.dataset_BJAir import get_dataloader, BJAir
# from data.dataset_London import get_dataloader
# from data.dataset_IntelLab import get_dataloader, IntelLab
from data.dataset_BJAir import get_dataloader
from utils import _quantile_CRPS_with_missing, _rmse_with_missing, _mae_with_missing, _mape_with_missing
import torch
import argparse
import datetime


def calucate(opt, save_dir):
    path = "config/" + opt.config
    torch.manual_seed(opt.seed)

    #save_dir = "save/" +opt.dataset +"/" + opt.model_path + opt.pred_attr
    # 检查文件是否存在，若存在则加载
    y = torch.load(save_dir + '/y_tensor.pt')
    label = torch.load(save_dir + '/label_tensor.pt')
    val_index = torch.load(save_dir + '/val_index_tensor.pt')
    time_utc = torch.load(save_dir + '/time_utc_tensor.pt')

    if opt.is_linear == 1:
        rmse = _rmse_with_missing(y.numpy(), label.numpy(), 1-val_index.numpy())
        mae = _mae_with_missing(y.numpy(), label.numpy(), 1-val_index.numpy())
        mape = _mape_with_missing(y.numpy(), label.numpy(), 1-val_index.numpy())
    else:
        rmse = _rmse_with_missing(y.median(dim=2, keepdim=True).values.numpy(), label.numpy(), 1 - val_index.numpy())
        mae = _mae_with_missing(y.median(dim=2, keepdim=True).values.numpy(), label.numpy(), 1 - val_index.numpy())
        mape = _mape_with_missing(y.median(dim=2, keepdim=True).values.numpy(), label.numpy(), 1 - val_index.numpy())

   # CPRS = _quantile_CRPS_with_missing(y, torch.cat([label.unsqueeze(-1)]*opt.nsample, dim=-1), 1-torch.cat([val_index.unsqueeze(-1)]*opt.nsample, dim=-1))

    print("----------------\n")
    print("mae:" ,mae)
    print("rmse:" ,rmse)
    print("mape:" ,mape)
    #print("CPRS:" ,CPRS)
    print()
    # Write metrics to "metric.txt"
    with open(save_dir + '/metric.txt', 'a') as f:
            f.write('\n  mae:  ' + str(mae))
            f.write('\n  rmse: ' + str(rmse))
            f.write('\n  mape: ' + str(mape))
           # f.write('\n  crps: ' + str(CPRS))


if __name__ == "__main__":
    # 获取当前日期和时间
    current_datetime = datetime.datetime.now().strftime("%Y%m%d%H%M")
    parser = argparse.ArgumentParser(description="DSTE")

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42, help='random seed for initialization')
    parser.add_argument("--config", type=str, default="base.yaml")
    parser.add_argument("--model_path", type=str, default="202501180105")

    parser.add_argument('--dataset', type=str, default='BJAir')
    parser.add_argument('--pred_attr', type=str, default="PM25_Concentration", help='Which AQ attribute to infer')
    parser.add_argument('--phase', type=str, default="train")
    parser.add_argument('--t_len', type=int, default=24, help='time window for inference')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--num_train_target', type=int, default=3, help='# of target set for training at the start')

    parser.add_argument('--nsample', type=int, default=50)

    parser.add_argument('--is_linear', type=int, default=1)
    parser.add_argument('--is_imputation', type=int, default=1)
    parser.add_argument('--need_attn', type=int, default=1)

    opt = parser.parse_args()
    calucate(opt)
         