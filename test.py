# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 13:55:00 2023

@author: LUN076
"""
from memory_profiler import profile
from calculate_loss import calucate
# from data.dataset_BJAir import get_dataloader, BJAir
# from data.dataset_London import get_dataloader
# from data.dataset_IntelLab import get_dataloader, IntelLab
from data.dataset_BJAir import get_dataloader
from torch.utils.data import DataLoader

from model.main_model import DSTE_BJAir
from utils import _quantile_CRPS_with_missing, _rmse_with_missing, _mae_with_missing, _mape_with_missing
import torch
import yaml
import argparse
from tqdm import tqdm
import datetime


def test(model, opt, test_loader, save_dir):
    model.eval()
    y = []
    label = []
    median_total = []
    val_index = []
    time_utc = []
    with torch.no_grad():
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it):
                val_pos, label_t, output = model.evaluate(test_batch, opt.nsample)
                # output = -output.permute(0, 2, 3, 1)
                y.append(output)
                label.append(label_t)
                val_index.append(val_pos.squeeze(-1))  # test_batch["pred_target"]
                time_utc.append(test_batch["time"])

    y = torch.cat(y).cpu()*opt.scale + opt.mean
    label = torch.cat(label).cpu()*opt.scale + opt.mean
    val_index = torch.cat(val_index).cpu()
    time_utc = torch.cat(time_utc).cpu()

    torch.save(y.clone().detach(), save_dir + '/y_tensor.pt')
    # label = label.squeeze(-1)
    val_index = val_index.cpu()
    torch.save(label.clone().detach(), save_dir + '/label_tensor.pt')
    torch.save(val_index.clone().detach(), save_dir + '/val_index_tensor.pt')
    torch.save(time_utc.clone().detach(), save_dir + '/time_utc_tensor.pt')

if __name__ == "__main__":
    # 获取当前日期和时间
    current_datetime = datetime.datetime.now().strftime("%Y%m%d%H%M")
    parser = argparse.ArgumentParser(description="DSTE")

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42, help='random seed for initialization')
    parser.add_argument("--config", type=str, default="base.yaml")
    parser.add_argument("--model_path", type=str, default="202501210756")

    parser.add_argument('--dataset', type=str, default='BJAir')
    parser.add_argument('--pred_attr', type=str, default="PM25_Concentration" ,  help='Which AQ attribute to infer')
    parser.add_argument('--phase', type=str, default="test") # dataset_base内有bug
    parser.add_argument('--t_len', type=int, default=24, help='time window for inference')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--num_train_target', type=int, default=3, help='# of target set for training at the start')
    
    parser.add_argument('--nsample', type=int, default=20)

    parser.add_argument('--is_linear', type=int, default=0, help='0扩散模型 1线性模型')
    parser.add_argument('--is_imputation', type=int, default=1, help='0外推任务 1插补任务')
    parser.add_argument('--is_neural_process', type=int, default=0, help='0正常loss 1神经过程loss')
    parser.add_argument('--need_attn', type=int, default=1)
    parser.add_argument('--missing_rate', type=float, default=0.3, help='插补任务的缺失率')

    opt = parser.parse_args()
    save_dir = "save/" + opt.dataset + "/" + opt.model_path + opt.pred_attr

    path = save_dir + "/" + opt.config
    torch.manual_seed(opt.seed)
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    config['is_linear'] = opt.is_linear
    config['is_imputation'] = opt.is_imputation
    config['need_attn'] = opt.need_attn
    config['num_train_target'] = opt.num_train_target
    config['is_neural_process'] = opt.is_neural_process
    config['missing_rate'] = opt.missing_rate
    #config["is_linear"] = 0 # 0扩散模型
    train_loader, valid_loader, test_loader = get_dataloader(opt)


    model = DSTE_BJAir(config, opt.device).to(opt.device)

    model.load_state_dict(torch.load(save_dir + "/model_latest.pth"), strict=False)
    model = model.to(opt.device)
    test(model, opt, test_loader, save_dir)
    calucate(opt, save_dir)
         