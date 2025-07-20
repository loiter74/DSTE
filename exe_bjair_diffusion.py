# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 15:06:28 2023

@author: LUN076
"""

from model.diffusion_model import DiffusionBase
from data.dataset_factory import get_dataloader
import torch
from tqdm import tqdm
import yaml
import argparse
import datetime
import os
from utils import get_info, _quantile_CRPS_with_missing

def train(model, optimizer, train_loader, opt, epoch, writer=None):
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
    total_diff_loss = 0.0
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
            pred_noise, noise, mask_impute, np_side_info, var, q_dists, p_dists = model(
                side_context,
                pred_context,
                side_target,
                pred_target,
                A,
                context_missing,
                missing_ratio=opt.missing_ratio  # 应用mask_impute
            )

            # 计算损失
            loss, mse_loss, reg_loss = model.compute_loss(
                pred_noise, noise,
                target_missing,
                mask_impute,
            )

            nll_loss, kl_loss = model.compute_np_loss(pred_target, np_side_info, var, q_dists, p_dists, target_missing)

            # 反向传播
            loss.backward()
            optimizer.step()
            # 更新统计量
            total_loss += loss.item()
            total_diff_loss += mse_loss.item()
            total_nll_loss += nll_loss.item()
            total_kl_loss += kl_loss.item()

            # 实时更新进度条显示
            avg_loss = total_loss / (batch_idx + 1)
            avg_diff_loss = total_diff_loss / (batch_idx + 1)
            avg_nll_loss = total_nll_loss / (batch_idx + 1)
            avg_kl_loss = total_kl_loss / (batch_idx + 1)
            pbar.set_postfix({
                'df': f"{avg_diff_loss:.2f}",
                'l': f"{avg_loss:.2f}",
                'kl': f"{avg_kl_loss:.2f}",
                'nl': f"{avg_nll_loss:.2f}",
            })

            if writer is not None:
                writer.add_scalar("avg_diff_loss", avg_diff_loss, epoch)
                writer.add_scalar("avg_loss", avg_loss, epoch)
                writer.add_scalar("avg_nll_loss", avg_nll_loss, epoch)
                writer.add_scalar("avg_kl_loss", avg_kl_loss, epoch)

    return total_loss / len(train_loader)


def validate(model, val_loader, opt):
    """
    带进度条的线性模型验证epoch
    Args:
        model: 线性预测模型
        val_loader: 验证数据加载器
        device: 计算设备
        epoch: 当前epoch索引（可选）
        total_epochs: 总epoch数（可选）
        missing_ratio: 缺失比例
    Returns:
        avg_loss: 平均损失
    """

    y_pred = []
    label = []
    mask = []
    mask_imputes = []

    pred_samples = []
    with torch.no_grad():
        with tqdm(val_loader, unit="batch", desc="Validation") as pbar:
            for batch_idx, batch in enumerate(pbar):
                # 数据预处理（与训练保持完全一致）
                # pred_context = batch['pred_context'].transpose(2, 3).to(opt.device).float()
                # side_context = batch['feat_context'].transpose(2, 3).to(opt.device).float()
                # pred_target = batch['pred_target'].transpose(2, 3).to(opt.device).float()
                # side_target = batch['feat_target'].transpose(2, 3).to(opt.device).float()
                # A = batch['adj_tc'].to(opt.device).float()
                # context_missing = batch['missing_mask_context'].transpose(1, 2).to(opt.device).float()
                # target_missing = batch['missing_mask_target'].transpose(1, 2).to(opt.device).float()

                # 数据预处理
                pred_context = batch['pred_context'].permute(0, 1, 3, 2).to(opt.device)
                side_context = batch['feat_context'].permute(0, 1, 3, 2).to(opt.device)
                pred_target = batch['pred_target'].permute(0, 1, 3, 2).to(opt.device)
                side_target = batch['feat_target'].permute(0, 1, 3, 2).to(opt.device)
                A = batch['adj_tc'].to(opt.device)
                context_missing = batch['missing_mask_context'].squeeze(-1).permute(0, 2, 1).to(opt.device)
                target_missing = batch['missing_mask_target'].squeeze(-1).permute(0, 2, 1).to(opt.device)

                samples, mask_impute = model.impute(
                    side_context, pred_context, side_target, pred_target,
                    A, context_missing, missing_ratio=opt.missing_ratio
                )
                pred_samples.append(samples)

                y_pred.append(samples.median(dim=1).values)
                label.append(pred_target.detach())
                mask_imputes.append(mask_impute)
                mask.append(
                        target_missing.clone().detach()
                        .permute(0, 2, 1)
                        .unsqueeze(2)
                        .contiguous())

                if batch_idx > 0: break
    return y_pred, label, mask, mask_imputes, pred_samples


if __name__ == "__main__":
    
    current_datetime = datetime.datetime.now().strftime("%Y%m%d%H%M")
    parser = argparse.ArgumentParser(description="DSTE")
    parser.add_argument("--config", type=str, default="base.yaml")
    parser.add_argument("--model_path", type=str, default="model_202505181849.pth", help="current_datetime")
    
    parser.add_argument('--pred_attr', type=str, default="PM25" ,  help='Which AQ attribute to infer')
    parser.add_argument('--t_len', type=int, default=24, help='time window for inference')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
    parser.add_argument('--num_train_target', type=int, default=3, help='# of target set for training at the start')
    parser.add_argument('--dataset_name', type=str, default="BJAir2017")

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--seed', type=int, default=42, help='random seed for initialization')
    parser.add_argument('--nsample', type=int, default=10)

    parser.add_argument("--missing_ratio", type=float, default=1.0, help="missing ratio")


    opt = parser.parse_args()
    path = "config/" + opt.config
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    train_loader, valid_loader, test_loader, scale, mean, var = get_dataloader(opt)
    opt.scale = scale
    opt.mean = mean
    opt.var = var
    torch.manual_seed(opt.seed)

    np_pretrained = "save/np/BJAir2017/PM25/model_202505182244.pth"
    # 阶段1：预训练神经过程模块
    diffusion_model = DiffusionBase(config, opt.device, np_pretrained_dir=np_pretrained).to(opt.device)
    # 检查是否提供了模型路径
    save_dir = "save/ddpm/" + opt.dataset_name + "/" + opt.pred_attr + "/"

    description = ""
    print(opt)
    if opt.model_path and os.path.exists(save_dir):
        print(f"Loading model from {opt.model_path}...")
        diffusion_model.load_state_dict(torch.load(save_dir + opt.model_path, map_location=opt.device, weights_only=False))
    else:
        print("No pre-trained model provided or path does not exist. Initializing a new model...")
        #optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=1e-4)
        optimizer = torch.optim.Adam(
            [p for n, p in diffusion_model.named_parameters() if "np_model" not in n],
            lr=opt.lr
        )
        #description = input("请输入一段关于训练的描述信息: ")

        epochs = 10
        # 使用示例
        for epoch in range(opt.epochs):
            epoch_loss = train(
                model=diffusion_model,
                optimizer=optimizer,
                train_loader=train_loader,
                opt=opt,
                epoch=epoch,
            )

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"created file {save_dir}")
        torch.save(diffusion_model.state_dict(), save_dir+ "model_" + current_datetime + description+ ".pth")

    print("missing_ratio: {:.2f} ".format(opt.missing_ratio))
    print(description)
    print("分布内")
    y_pred, label, mask, mask_imputes, pred_samples = validate(diffusion_model, valid_loader, opt)
    get_info(opt, y_pred, label, mask, mask_imputes)


    for i in range(len(mask)):
        mask[i] =  (1-mask[i])*(1-mask_imputes[i].unsqueeze(2).to(opt.device))
    CRPS = _quantile_CRPS_with_missing(pred_samples, label, mask)
    print("CRPS: ", CRPS)


    # print("分布外")
    # y_pred, label, mask, mask_imputes, pred_samples = validate(diffusion_model, test_loader, opt)
    # get_info(opt, y_pred, label, mask, mask_imputes)
    # for i in range(len(mask)):
    #     mask[i] = (1 - mask[i]) * (1 - mask_imputes[i].unsqueeze(2).to(opt.device))
    # CRPS = _quantile_CRPS_with_missing(pred_samples, label, mask)
    #
    # print("CRPS: ", CRPS)
    #
    # from data.dataset_London import get_dataloader
    # _, _, test_loader = get_dataloader(opt)
    #
    # print("LD分布外")
    # y_pred, label, mask, mask_imputes, pred_samples = validate(diffusion_model, test_loader, opt)
    # get_info(opt, y_pred, label, mask, mask_imputes)
    # for i in range(len(mask)):
    #     mask[i] =  (1-mask[i])*(1-mask_imputes[i].unsqueeze(2).to(opt.device))
    # CRPS = _quantile_CRPS_with_missing(pred_samples, label, mask)

    # print("CRPS: ", CRPS)
