# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.distributions import kl_divergence

from model.inner.deter_layer import Deterministic
from model.inner.inference_layer import InferenceModel
import torch.nn.functional as F

from model.inner.observation_layer import ObservationModel


class NeuralProcessBase(nn.Module):
    """神经过程基础模块"""
    def __init__(self, config):
        super().__init__()
        # 确定性特征提取器
        self.deter = Deterministic(
            input_dim=config['input_dim'],
            covariate_dim=7,
            emd_channel=config['emd_channel'],
            tcn_channels=config['tcn_channels'],
            tcn_kernel_size=config['tcn_kernel_size'],
            dropout=config['dropout']
        )

        # 推理模型
        self.infer = InferenceModel(
            tcn_channels=config['tcn_channels'],
            latent_channels=config['latent_channels'],
            num_hidden_layers=config['num_latent_layers']
        )

        #
        self.observation = ObservationModel(sum(config['latent_channels']), config['input_dim'],
                                            config['observation_hidden_dim'], config['num_observation_layers'])

    def forward(self, x_context, y_context, x_target, y_target, adj, missing_index, noise=None):
        """神经过程前向计算"""
        if noise is None and y_target is not None:
            noise = torch.randn_like(y_target)
        d_c, d_t = self.deter(
            x_context=x_context,
            y_context=y_context,
            x_target=x_target,
            y_target=noise,
            adj=adj,
            missing_mask_context=missing_index
        )
        q_target, q_dists = self.infer(
            d_c=d_c,
            d_t=d_t,
            adj=adj[:, 0],  # 取静态邻接矩阵
            missing_index_context=missing_index
        )

        if y_target is not None:
            d_c, d_t = self.deter(
                x_context=x_context,
                y_context=y_context,
                x_target=x_target,
                y_target=y_target,
                adj=adj,
                missing_mask_context=missing_index
            )
            p_target, p_dists = self.infer(
                d_c=d_c,
                d_t=d_t,
                adj=adj[:, 0],  # 取静态邻接矩阵
                missing_index_context=missing_index
            )
        else:
            p_target, p_dists = q_target, q_dists

        mu, var = self.observation(p_target)
        return mu, var, q_dists, p_dists

    def compute_kl(self, q_dists, p_dists, valid_index):
        """计算所有层次的条件KL散度"""

    @staticmethod
    def compute_np_loss(y, pred_mu, pred_var, p_dists, q_dists, valid_mask):
        """计算该模块的loss分量"""
        # 预测损失
        valid_index = 1-valid_mask.unsqueeze(2).permute(0,3,2,1)

        nll = 0.5 * ((y - pred_mu) ** 2 / pred_var + torch.log(pred_var))

        #y_loss = F.mse_loss(pred, y) # , reduction='none'
        nll_loss = (nll * valid_index).sum() / valid_index.sum()
        # KL散度
        kl = []
        for q, p in zip(q_dists, p_dists):
            kl.append(kl_divergence(q, p))
        kl_loss = (torch.cat(kl, dim=2) * valid_index).sum() / valid_index.sum()

        return nll_loss, kl_loss

    @staticmethod
    def compute_mae_loss(y, pred_mu, valid_mask):
        """计算该模块的loss分量"""
        # 预测损失
        valid_index = 1-valid_mask.unsqueeze(2).permute(0,3,2,1)
        loss = F.mse_loss(pred_mu, y) # , reduction='none'
        loss = (loss * valid_index).sum() / valid_index.sum()

        return loss