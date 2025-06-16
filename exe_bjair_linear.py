# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 15:06:28 2023

@author: LUN076
"""

from model.linear_model import ImLinearBase
from data.dataset_BJAir2017 import get_dataloader
#from data.dataset_PEMS03_small import get_dataloader
import torch
from tqdm import tqdm
import yaml
import argparse
import datetime
from utils import get_info, generate_missing_mask


def train(model, optimizer, train_loader, opt, epoch):
    """
    带进度条的线性模型训练epoch
    Args:
        model: 线性预测模型
        optimizer: 优化器
        train_loader: 训练数据加载器
        device: 计算设备
        epoch: 当前epoch索引
        total_epochs: 总epoch数
        missing_ratio: 缺失比例
    Returns:
        avg_loss: 平均损失
    """
    model.train()
    total_loss = 0.0

    with tqdm(train_loader, unit="batch", desc="epoch: "+str(epoch)) as pbar:
        for batch_idx, batch in enumerate(pbar):
            # 数据预处理
            pred_context = batch['pred_context'].transpose(2, 3).to(opt.device).float()
            side_context = batch['feat_context'].transpose(2, 3).to(opt.device).float()
            pred_target = batch['pred_target'].transpose(2, 3).to(opt.device).float()
            side_target = batch['feat_target'].transpose(2, 3).to(opt.device).float()
            A = batch['adj'].to(opt.device).float()
            context_missing = batch['missing_mask_context'].transpose(1, 2).to(opt.device).float()
            target_missing = batch['missing_mask_target'].transpose(1, 2).to(opt.device).float()

            # 梯度清零
            optimizer.zero_grad()
            # 前向传播
            mask0 = generate_missing_mask(pred_target.shape, missing_ratio=opt.missing_ratio).to(opt.device).float()

            y = model(
                side_context,
                pred_context,
                side_target,
                pred_target*mask0,  # 应用maskNone
                A,
                context_missing,
                mask0
            )

            # 计算损失
            loss, mse_loss, reg_loss = model.compute_loss_impute(
                y,
                pred_target,
                target_missing,
                mask0
            )

            # 反向传播
            loss.backward()
            optimizer.step()
            # 更新统计量
            total_loss += loss.item()

            # 实时更新进度条显示
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({
                'loss': f"{avg_loss:.4f}",
                # 'mse_loss': f"{mse_loss.item():.4f}",
                # 'reg_loss': f"{reg_loss.item():.4f}",
            })

    return total_loss / len(train_loader)


def validate(model, val_loader, opt):
    model.eval()

    y_pred = []
    label = []
    mask = []

    mask_impute = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            # 数据预处理（与训练一致）
            pred_context = batch['pred_context'].transpose(2, 3).to(opt.device).float()
            side_context = batch['feat_context'].transpose(2, 3).to(opt.device).float()
            pred_target = batch['pred_target'].transpose(2, 3).to(opt.device).float()
            side_target = batch['feat_target'].transpose(2, 3).to(opt.device).float()
            A = batch['adj'].to(opt.device).float()
            context_missing = batch['missing_mask_context'].transpose(1, 2).to(opt.device).float()
            target_missing = batch['missing_mask_target'].transpose(1, 2).to(opt.device).float()

            mask0 = generate_missing_mask(pred_target.shape, missing_ratio=opt.missing_ratio).to(opt.device)

            # 前向计算
            y = model(
                side_context, pred_context,
                side_target, pred_target*mask0,
                A, context_missing,
                mask0
            )
            y_pred.append(y.clone().detach())
            label.append(pred_target.detach())
            mask.append(
                target_missing.clone().detach()
                .permute(0, 2, 1)
                .unsqueeze(2)
                .contiguous())
            mask_impute.append(mask0)

    return y_pred, label, mask, mask_impute


if __name__ == "__main__":
    
    current_datetime = datetime.datetime.now().strftime("%Y%m%d%H%M")
    parser = argparse.ArgumentParser(description="DSTE")
    parser.add_argument("--config", type=str, default="base.yaml")
    parser.add_argument("--model_path", type=str, default="")
    
    parser.add_argument('--pred_attr', type=str, default="PM25_Concentration" ,  help='Which AQ attribute to infer')
    parser.add_argument('--t_len', type=int, default=24, help='time window for inference')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
    parser.add_argument('--num_train_target', type=int, default=3, help='# of target set for training at the start')
    parser.add_argument('--dataset', type=str, default="BJAir")

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--seed', type=int, default=42, help='random seed for initialization')
    parser.add_argument('--nsample', type=int, default=20)
   # parser.add_argument('--missing_rate', type=float, default=0.6)
    parser.add_argument("--missing_ratio", type=float, default=0.7, help="missing ratio")

    opt = parser.parse_args()
    path = "config/" + opt.config
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    config = config["configLinear"]

    train_loader, valid_loader, test_loader = get_dataloader(opt)
    torch.manual_seed(opt.seed)

    # TODO: 加载预训练的 DSTE（im_diff 版本）状态字典 预训练
    # TODO: 创建 DSTE 模型并转移权重
    linear_model = ImLinearBase(config).to(opt.device)
    optimizer = torch.optim.Adam(linear_model.parameters(), lr=1e-4)

    epochs = 30
    # 使用示例
    for epoch in range(epochs):
        epoch_loss = train(
            model=linear_model,
            optimizer=optimizer,
            train_loader=train_loader,
            opt=opt,
            epoch=epoch,
        )

    y_pred, label, mask, mask_impute = validate(linear_model, valid_loader, opt)

    get_info(opt, y_pred, label, mask, mask_impute)
    print("missing_ratio: {:.2f} ".format(opt.missing_ratio))


            