# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 15:06:28 2023

@author: LUN076
"""
from calculate_loss import calucate
from model.main_model import DSTE_BJAir, DSTE_BjAir2017
from data.dataset_BJAir2017 import get_dataloader
#from data.dataset_PEMS03 import get_dataloader
import torch
from torch.optim import Adam
from tqdm import tqdm
import yaml
import argparse
import datetime
import os
import json

from test import test

if __name__ == "__main__":
    
    current_datetime = datetime.datetime.now().strftime("%Y%m%d%H%M")
    parser = argparse.ArgumentParser(description="DSTE")
    parser.add_argument("--config", type=str, default="base.yaml")
    parser.add_argument("--model_path", type=str, default="")
    
    parser.add_argument('--pred_attr', type=str, default="PM25_Concentration" ,  help='Which AQ attribute to infer')
    parser.add_argument('--t_len', type=int, default=24, help='time window for inference')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
    parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
    parser.add_argument('--num_train_target', type=int, default=3, help='# of target set for training at the start')
    parser.add_argument('--dataset', type=str, default="BJAir")

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--seed', type=int, default=42, help='random seed for initialization')
    parser.add_argument('--nsample', type=int, default=20)

    # 添加一个新的参数来表示模型名称
    parser.add_argument('--model_name', type=str, default='扩散外推', help='Model name')
    parser.add_argument('--is_linear', type=int, default=0, help='0扩散模型 1线性模型')
    parser.add_argument('--is_imputation', type=int, default=1, help='0外推任务 1插补任务')
    parser.add_argument('--missing_rate', type=float, default=1, help='插补任务的缺失率')
    parser.add_argument('--is_neural_process', type=int, default=0, help='0正常loss 1神经过程loss')
    parser.add_argument('--need_attn', type=int, default=1)
   # parser.add_argument('--missing_rate', type=float, default=0.6)

    opt = parser.parse_args()
    path = "config/" + opt.config
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    config['is_linear'] = opt.is_linear
    config['is_imputation'] = opt.is_imputation
    config['need_attn'] = opt.need_attn
    config['num_train_target'] = opt.num_train_target
    config['is_neural_process'] = opt.is_neural_process
    config['missing_rate'] = opt.missing_rate

    train_loader, valid_loader, test_loader = get_dataloader(opt)
    torch.manual_seed(opt.seed)

    # TODO: 加载预训练的 DSTE（im_diff 版本）状态字典 预训练
    # TODO: 创建 DSTE 模型并转移权重

    model = DSTE_BjAir2017(config, opt.device).to(opt.device)
    if opt.model_path != "":
        if os.path.isfile(opt.model_path):
            if torch.load(opt.model_path) is dict:
                model.load_state_dict(torch.load(opt.model_path))
            else:
                loaded_model = torch.load(opt.model_path)
                model = DSTE_BJAir(config, opt.device).to(opt.device)
                model.load_state_dict(loaded_model.state_dict())
        print("load model has trained")

    save_dir = "save/" +opt.dataset +"/" + current_datetime+opt.pred_attr
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        opt_dict = vars(opt)
        print(opt)
        with open(save_dir+'/opt.txt', 'w') as f:
            json.dump(opt_dict, f, indent=4)
            # 将 config 内容保存到文件
        with open(save_dir + '/base.yaml', 'w') as f:
            yaml.dump(config, f)
        print(f"created file {save_dir}")

    description = input("请输入一段关于训练的描述信息: ")
    with open(save_dir + '/description.txt', 'w') as f:
        f.write(description)

    optimizer = Adam(model.parameters(), lr=opt.lr, weight_decay=1e-4)
    
    p1 = int(0.5 * opt.epochs)
    p2 = int(0.75 * opt.epochs)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[p1, p2], gamma=0.1)

    no_improvement_count = 0  # 新增：记录未更新最佳验证损失的次数
    best_valid_loss = float('inf')  # 初始最佳验证损失为正无穷

    for epoch_no in range(opt.epochs):
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=1.0, maxinterval=100.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()
                loss = model(train_batch)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "loss": avg_loss,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
        lr_scheduler.step()
        
        if valid_loader is not None and (epoch_no + 1) % 30 == 0 :
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        loss = model(valid_batch, is_train=False)
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "valid_loss": avg_loss_valid,
                                "epoch": epoch_no,
                            },
                            refresh=True,
                        )
            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                file_path = save_dir+'/model_best.pth'
                torch.save(model.state_dict(), file_path)

                print(
                    "\n best loss is updated to ",
                    avg_loss_valid / batch_no,
                    "at",
                    epoch_no,
                )
                no_improvement_count = 0  # 重置计数器
            else:
                no_improvement_count += 1  # 未更新最佳验证损失，计数器加 1

        if no_improvement_count >= 3:  # 新增：如果三次未更新最佳验证损失，终止训练
            print("No improvement in validation loss for 3 times. Early stopping.")
            break

    torch.save(model.state_dict(), save_dir+'/model_latest.pth')
    test(model, opt, test_loader, save_dir)
    calucate(opt, save_dir)
            