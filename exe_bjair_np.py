# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 15:06:28 2023

@author: LUN076
"""
from data.dataset_factory import get_dataloader
# from data.dataset_BJAir2017 import get_dataloader
#from data.dataset_PEMS03_small import get_dataloader

import torch
from tqdm import tqdm
import yaml
import argparse
import datetime
import os
import json
from model.np_model import NeuralProcessBase
from utils import get_info


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
    Returns:
        avg_loss: 平均损失
    """
    model.train()


    total_loss = 0.0
    total_nll_loss = 0.0
    total_kl_loss = 0.0
    with tqdm(train_loader, unit="batch", desc="epoch: "+str(epoch)) as pbar:
        for batch_idx, batch in enumerate(pbar):
            # 数据预处理
            pred_context = batch['pred_context'].permute(0, 1, 3, 2).to(opt.device)
            side_context = batch['feat_context'].permute(0, 1, 3, 2).to(opt.device)
            pred_target = batch['pred_target'].permute(0, 1, 3, 2).to(opt.device)
            side_target = batch['feat_target'].permute(0, 1, 3, 2).to(opt.device)
            A = batch['adj_tc'].to(opt.device)
            context_missing = batch['missing_mask_context'].squeeze(-1).permute(0, 2, 1).to(opt.device)
            target_missing = batch['missing_mask_target'].squeeze(-1).permute(0, 2, 1).to(opt.device)

            # 梯度清零
            optimizer.zero_grad()
            # 前向传播
            pred_mu, pred_var, q_dists, p_dists = model(
                side_context, pred_context,
                side_target, pred_target,
                A, context_missing
            )

            # 损失计算
            nll_loss, kl_loss = model.compute_np_loss(pred_target, pred_mu, pred_var, q_dists, p_dists, target_missing)

            # 反向传播
            loss = nll_loss +  kl_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            total_nll_loss += nll_loss.item()
            total_kl_loss += kl_loss.item()

            # 实时更新进度条显示
            avg_loss = total_loss / (batch_idx + 1)
            avg_nll_loss = total_nll_loss / (batch_idx + 1)
            avg_kl_loss = total_kl_loss / (batch_idx + 1)
            pbar.set_postfix({
                'l': f"{avg_loss:.2f}",
                'kl': f"{avg_kl_loss:.2f}",
                'nl': f"{avg_nll_loss:.2f}",
            })

        return total_loss / len(train_loader)

def validate(model, val_loader, opt):
    model.eval()

    y_pred = []
    label = []
    mask = []

    with torch.no_grad():
        with tqdm(val_loader, unit="batch", desc="Validation") as pbar:
            for batch_idx, batch in enumerate(pbar):
                # 数据预处理
                pred_context = batch['pred_context'].permute(0, 1, 3, 2).to(opt.device)
                side_context = batch['feat_context'].permute(0, 1, 3, 2).to(opt.device)
                pred_target = batch['pred_target'].permute(0, 1, 3, 2).to(opt.device)
                side_target = batch['feat_target'].permute(0, 1, 3, 2).to(opt.device)
                A = batch['adj_tc'].to(opt.device)
                context_missing = batch['missing_mask_context'].squeeze(-1).permute(0, 2, 1).to(opt.device)
                target_missing = batch['missing_mask_target'].squeeze(-1).permute(0, 2, 1).to(opt.device)

                # 前向计算
                pred_mu, pred_var, q_dists, p_dists = model(
                    side_context, pred_context,
                    side_target, None,
                    A, context_missing
                )
                y_pred.append(pred_mu)
                label.append(torch.tensor(pred_target))
                mask.append(torch.tensor(target_missing).permute(0,2,1).unsqueeze(2))

    return y_pred, label, mask

if __name__ == "__main__":

    current_datetime = datetime.datetime.now().strftime("%Y%m%d%H%M")
    parser = argparse.ArgumentParser(description="DSTE")
    parser.add_argument("--config", type=str, default="base.yaml")
    parser.add_argument("--model_path", type=str, default="")

    parser.add_argument('--pred_attr', type=str, default="PM25", help='Which AQ attribute to infer')
    parser.add_argument('--t_len', type=int, default=24, help='time window for inference')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
    parser.add_argument('--num_train_target', type=int, default=3, help='# of target set for training at the start')
    parser.add_argument('--dataset_name', type=str, default="BJAir2017")

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--seed', type=int, default=42, help='random seed for initialization')

    opt = parser.parse_args()
    path = "config/" + opt.config
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    config = config["configNP"]
    train_loader, valid_loader, test_loader, scale, mean, var = get_dataloader(opt)
    torch.manual_seed(opt.seed)

    # TODO: 加载预训练的 DSTE（im_diff 版本）状态字典 预训练
    # TODO: 创建 DSTE 模型并转移权重
    save_dir = "save/np/"  + opt.dataset_name + "/" + opt.pred_attr + "/"
    np_model = NeuralProcessBase(config).to(opt.device)
    description = None
    if opt.model_path and os.path.exists(save_dir):
        print(f"Loading model from {opt.model_path}...")
        np_model.load_state_dict(
            torch.load(save_dir + opt.model_path, map_location=opt.device, weights_only=False))

    else:
        print("No pre-trained model provided or path does not exist. Initializing a new model...")
        optimizer = torch.optim.Adam(np_model.parameters(), lr=1e-3)
        description = ""  # input("请输入一段关于训练的描述信息: ")

        # 使用示例
        for epoch in range(opt.epochs):
            epoch_loss = train(
                model=np_model,
                optimizer=optimizer,
                train_loader=train_loader,
                opt=opt,
                epoch=epoch,
            )

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"created file {save_dir}")
        torch.save(np_model.state_dict(), save_dir + "model_" + current_datetime + description + ".pth")

    print("分布内")
    y_pred, label, mask = validate(np_model, valid_loader, opt)
    get_info(opt, y_pred, label, mask, None)


    print("分布外")
    y_pred, label, mask = validate(np_model, test_loader, opt)
    get_info(opt, y_pred, label, mask, None)
    from data.dataset_London import get_dataloader
    _, _, test_loader = get_dataloader(opt)

    print("LD分布外")
    y_pred, label, mask = validate(np_model, test_loader, opt)

    get_info(opt, y_pred, label, mask, None)


