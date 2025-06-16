import argparse
import datetime

import torch
from tqdm import tqdm

from data.dataset_BJAir2017 import get_dataloader
from data.dataset_PEMS03_small import get_dataloader
from utils import generate_missing_mask, get_info


def validate(model, val_loader, device):
    y_pred = []
    label = []
    mask = []

    mask_impute = []

    for batch in tqdm(val_loader, desc="Validating"):
            # 数据预处理（与训练一致）
            pred_context = batch['pred_context'].transpose(2, 3).to(device).float()
            side_context = batch['feat_context'].transpose(2, 3).to(device).float()
            pred_target = batch['pred_target'].transpose(2, 3).to(device).float()
            side_target = batch['feat_target'].transpose(2, 3).to(device).float()
            A = batch['adj'].to(device).float()
            context_missing = batch['missing_mask_context'].transpose(1, 2).to(device).float()
            target_missing = batch['missing_mask_target'].transpose(1, 2).to(device).float()

            mask0 = generate_missing_mask(pred_target.shape, missing_ratio=0.3).to(device)
            y_unfill = pred_target*mask0

            mean = y_unfill.sum()/mask0.sum()
            y = y_unfill + mean*(1-mask0)

            y_pred.append(y.detach())
            label.append(pred_target.detach())
            mask.append(
                target_missing.clone().detach()
                .permute(0, 2, 1)
                .unsqueeze(2)
                .contiguous())
            mask_impute.append(mask0)

    return (torch.concat(y_pred,dim=0),
            torch.concat(label,dim=0),
            torch.concat(mask,dim=0),
            torch.concat(mask_impute,dim=0))


if __name__ == '__main__':
    current_datetime = datetime.datetime.now().strftime("%Y%m%d%H%M")
    parser = argparse.ArgumentParser(description="DSTE")

    parser.add_argument('--pred_attr', type=str, default="PM25_Concentration", help='Which AQ attribute to infer')
    parser.add_argument('--t_len', type=int, default=24, help='time window for inference')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
    parser.add_argument('--num_train_target', type=int, default=3, help='# of target set for training at the start')
    parser.add_argument('--dataset', type=str, default="BJAir")
    parser.add_argument('--seed', type=int, default=42, help='random seed for initialization')
    parser.add_argument('--device', type=str, default='cuda:0')

    opt = parser.parse_args()
    train_loader, valid_loader, test_loader = get_dataloader(opt)
    y_pred, label, mask, mask_impute = validate(None, valid_loader, opt.device)

    get_info(opt, y_pred, label, mask, mask_impute)