import numpy as np
import torch
import torch.nn as nn
from torch.distributions import kl_divergence

from model.contrast.dsti import DSTI
from model.inner.st_module.graph_layer import StaticAgg, DynamicAgg
from model.np_model import NeuralProcessBase
from utils import generate_missing_mask


class DiffusionBase(nn.Module):
    def __init__(self, config, device, np_pretrained_dir=None):
        super().__init__()
        self.config = config["configCSDI"]
        config_diff = config["configCSDI"]
        self.device = device
        self.n_samples = 2

        #self.diff_model = CSDI(config["configCSDI"])

        self.diff_model = DSTI(config["configCSDI"])
        self.np_model = NeuralProcessBase(config["configNP"])
        if np_pretrained_dir is not None:
            self.np_model.load_state_dict(torch.load(np_pretrained_dir))

        #self.diff_model = DSTE(config)
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)

        self.graph_aggeration = DynamicAgg(pred_in=1, feat_in=7, channels=128,
                                           out=1)
        self.graph_aggeration2 = StaticAgg(pred_in=1, channels=128, out_channels=1)


    def forward(self, x_context, y_context, x_target, y_target, adj, missing_mask, missing_ratio):
        #NP PART
        noise = torch.randn_like(y_target, device=self.device)
        np_side_info, var, q_dists, p_dists = self.np_model(x_context, y_context, x_target, y_target, adj, missing_mask, noise)

        # DDPM PART
        y_target = y_target.squeeze(2)
        noise = noise.squeeze(2)
        t = torch.randint(0, self.num_steps, [y_target.shape[0]]).to(self.device)
        current_alpha = self.alpha_torch[t]
        noisy_data = (current_alpha ** 0.5) * y_target + (1.0 - current_alpha) ** 0.5 * noise
        mask_impute = generate_missing_mask(y_target.shape, missing_ratio=missing_ratio, device=self.device) # 插补训练的掩码 1可见 0掩码

        noisy_data = (noisy_data* (1-mask_impute)).unsqueeze(1)
        obs_data =  (y_target*mask_impute).unsqueeze(1)

        graph_agg = self.graph_aggeration(x_target, x_context, y_context).permute(0, 2, 1, 3)
        graph_agg2 = self.graph_aggeration2(y_context, adj[:, 1]).permute(0, 2, 1, 3)

        # 图信息
        diff_input = torch.cat([noisy_data, obs_data, graph_agg2], dim=1)

        # 外推np信息
        diff_input = torch.cat([noisy_data, np_side_info.permute(0, 2, 1, 3), graph_agg2], dim=1)

        #side_info = torch.cat([st_info, np_side_info], dim=2)
        pred_noise = self.diff_model(diff_input, np_side_info, t)

        # 最终预测
        return pred_noise, noise, mask_impute, np_side_info, var, q_dists, p_dists

    @staticmethod
    def compute_np_loss(label, pred_mu, pred_var, p_dists, q_dists, valid_mask):
        """计算该模块的loss分量"""
        # 预测损失
        valid_index = 1-valid_mask.unsqueeze(2).permute(0,3,2,1)
        nll = 0.5 * ((label - pred_mu) ** 2 / pred_var + torch.log(pred_var))
        nll_loss = (nll * valid_index).sum() / valid_index.sum()
        # KL散度
        kl = []
        for q, p in zip(q_dists, p_dists):
            kl.append(kl_divergence(q, p))
        kl_loss = (torch.cat(kl, dim=2) * valid_index).sum() / valid_index.sum()

        return nll_loss, kl_loss

    def compute_loss(self, pred_noise, target_noise, mask_true, mask_impute):

        mse_loss = ((((pred_noise - target_noise)*(1-mask_impute)) ** 2)).sum()/(1-mask_impute).sum()
        # 对observation模块的所有参数添加L2正则
        reg_loss = 0.0
        for param in self.parameters():
            reg_loss += torch.norm(param, p=2)  # L2正则
        return mse_loss , mse_loss, reg_loss  # 调节0.01为合适系数+ 0.001 * reg_loss


    def impute(self, side_context,
                pred_context,
                side_target,
                pred_target,
                A,
                context_missing,
                missing_ratio,
                n_samples=20):


        #st_info , _ , _, _ = self.ST_model(side_context, pred_context, side_target, pred_target, A, context_missing)

        #side_info = torch.cat([st_info, np_side_info], dim=2)
        graph_agg = self.graph_aggeration(side_target, side_context, pred_context).permute(0, 2, 1, 3)
        graph_agg2 = self.graph_aggeration2(pred_context, A[:, 1]).permute(0, 2, 1, 3)


        pred_target = pred_target.squeeze(2)
        # generate noisy observation for unconditional model
        mask_impute = generate_missing_mask(pred_target.shape, missing_ratio=missing_ratio,
                                            device=self.device)  # 插补训练的掩码 1可见 0掩码
        B, K, L = pred_target.shape
        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)
        for i in range(n_samples):

            current_sample = torch.randn_like(pred_target)
            # NP info
            np_side_info, var, q_dists, p_dists = self.np_model(side_context,
                                                                pred_context,
                                                                side_target,
                                                                None,
                                                                A,
                                                                context_missing,
                                                                current_sample.unsqueeze(2))
            for t in range(self.num_steps - 1, -1, -1):
                cond_obs = (mask_impute * pred_target).unsqueeze(1)
                noisy_target = ((1 - mask_impute) * current_sample).unsqueeze(1)

                # np外推
                diff_input = torch.cat([noisy_target, np_side_info.permute(0, 2, 1, 3), graph_agg2], dim=1)  # (B,2,K,L)
                #diff_input = torch.cat([noisy_target, cond_obs, graph_agg2], dim=1)  # (B,2,K,L)

                # 无图信息
                #diff_input = torch.cat([noisy_target, cond_obs, graph_agg2], dim=1)  # (B,2,K,L)

                predicted = self.diff_model(diff_input, np_side_info, torch.tensor([t]).to(self.device))

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise

            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples, mask_impute

