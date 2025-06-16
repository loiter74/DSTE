import numpy as np
import torch
import torch.nn as nn
from torch.distributions import kl_divergence

from model.contrast.csdi import CSDI
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

        self.diff_model = CSDI(config["configCSDI"])

        #self.diff_model = DSTI(config["configCSDI"])
        self.np_model = NeuralProcessBase(config["configNP"])
        if np_pretrained_dir is not None:
            self.np_model.load_state_dict(torch.load(np_pretrained_dir))
            # 冻结 np_model 的所有参数
            for param in self.np_model.parameters():
                param.requires_grad = False

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

        # DDIM相关参数
        self.ddim_sampling = config_diff.get("ddim_sampling", False)  # 是否使用DDIM采样
        self.ddim_eta = config_diff.get("ddim_eta", 0.0)  # DDIM噪声参数，0为完全确定性
        self.ddim_steps = config_diff.get("ddim_steps", 50)  # DDIM采样步数，通常小于DDPM步数
        self.ddim_discr = config_diff.get("ddim_discr", "uniform")  # 采样步骤选择方式

        # 预计算DDIM采样时间步
        if self.ddim_sampling:
            self.ddim_timesteps = self._get_ddim_timesteps()
            self.ddim_alphas = self.alpha[self.ddim_timesteps]
            self.ddim_alphas_prev = np.append(1.0, self.ddim_alphas[:-1])
            self.ddim_sigmas = self.ddim_eta * np.sqrt(
                (1 - self.ddim_alphas_prev) / (1 - self.ddim_alphas) * (1 - self.ddim_alphas / self.ddim_alphas_prev)
            )

    def _get_ddim_timesteps(self):
        """
        根据设置的采样方式选择DDIM时间步
        """
        if self.ddim_discr == "uniform":
            c = self.num_steps // self.ddim_steps
            ddim_timesteps = np.asarray(list(range(0, self.num_steps, c)))
        elif self.ddim_discr == "quad":
            ddim_timesteps = ((np.linspace(0, np.sqrt(self.num_steps * 0.8), self.ddim_steps)) ** 2).astype(int)
        else:
            raise NotImplementedError(f"Unknown discretization method: {self.ddim_discr}")

        # 确保时间步是降序的(从T到0)
        ddim_timesteps = np.sort(ddim_timesteps)[::-1]
        return ddim_timesteps

    def forward(self, x_context, y_context, x_target, y_target, adj, missing_mask, missing_ratio):
        # 原有的forward方法保持不变
        noise = torch.randn_like(y_target, device=self.device)
        np_side_info, var, q_dists, p_dists = self.np_model(x_context, y_context, x_target, y_target, adj, missing_mask,
                                                            noise)

        # DDPM PART
        y_target = y_target.squeeze(2)
        noise = noise.squeeze(2)
        t = torch.randint(0, self.num_steps, [y_target.shape[0]]).to(self.device)
        current_alpha = self.alpha_torch[t]
        noisy_data = (current_alpha ** 0.5) * y_target + (1.0 - current_alpha) ** 0.5 * noise
        mask_impute = generate_missing_mask(y_target.shape, missing_ratio=missing_ratio, device=self.device)

        noisy_data = (noisy_data * (1 - mask_impute)).unsqueeze(1)
        obs_data = (y_target * mask_impute).unsqueeze(1)

        graph_agg_d = self.graph_aggeration(x_target, x_context, y_context).permute(0, 2, 1, 3)
        graph_agg_s = self.graph_aggeration2(y_context, adj[:, 1]).permute(0, 2, 1, 3)

        # stand
        # b c n t
        #diff_input = torch.cat([noisy_data, obs_data, graph_agg_d, graph_agg_s], dim=1)
        diff_input = torch.cat([noisy_data, obs_data, noisy_data, obs_data], dim=1)

        pred_noise = self.diff_model(diff_input, np_side_info, t)

        return pred_noise, noise, mask_impute, np_side_info, var, q_dists, p_dists

    def ddim_impute(self, side_context, pred_context, side_target, pred_target, A, context_missing, missing_ratio,
                    n_samples=1):
        """
        使用DDIM进行加速采样
        """
        pred_target = pred_target.squeeze(2)
        mask_impute = generate_missing_mask(pred_target.shape, missing_ratio=missing_ratio, device=self.device)

        B, K, L = pred_target.shape
        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        graph_agg_d = self.graph_aggeration(side_target, side_context, pred_context).permute(0, 2, 1, 3)
        graph_agg_s = self.graph_aggeration2(pred_context, A[:, 1]).permute(0, 2, 1, 3)

        for i in range(n_samples):
            current_sample = torch.randn_like(pred_target)

            np_side_info, var, q_dists, p_dists = self.np_model(
                side_context, pred_context, side_target, None, A, context_missing, current_sample.unsqueeze(2)
            )

            for idx, t in enumerate(self.ddim_timesteps):
                # 批量处理时间步
                t_tensor = torch.full((B,), t, device=self.device, dtype=torch.long)

                cond_obs = (mask_impute * pred_target).unsqueeze(1)
                noisy_target = ((1 - mask_impute) * current_sample).unsqueeze(1)
                diff_input = torch.cat([noisy_target, cond_obs, graph_agg_d, graph_agg_s], dim=1)

                predicted_noise = self.diff_model(diff_input, np_side_info, t_tensor).squeeze(1)

                # 使用torch计算DDIM参数
                alpha_t = torch.tensor(self.alpha[t], device=self.device).float()
                alpha_prev = torch.tensor(self.ddim_alphas_prev[idx], device=self.device).float()
                sigma_t = torch.tensor(self.ddim_sigmas[idx], device=self.device).float()

                # 预测x_0
                pred_x0 = (current_sample - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)

                # 下一步采样
                if idx < len(self.ddim_timesteps) - 1:
                    if self.ddim_eta > 0:
                        noise = torch.randn_like(current_sample)
                    else:
                        noise = torch.zeros_like(current_sample)

                    current_sample = torch.sqrt(alpha_prev) * pred_x0 + \
                                     torch.sqrt(1 - alpha_prev - sigma_t ** 2) * predicted_noise + \
                                     sigma_t * noise
                else:
                    current_sample = pred_x0

            imputed_samples[:, i] = current_sample.detach()

        return imputed_samples, mask_impute

    def impute(self, side_context, pred_context, side_target, pred_target, A, context_missing, missing_ratio,
               n_samples=40):
        """
        根据配置选择DDPM或DDIM进行采样
        """
        if self.ddim_sampling:
            return self.ddim_impute(side_context, pred_context, side_target, pred_target, A, context_missing,
                                    missing_ratio, n_samples)

        # 原有的DDPM采样代码
        graph_agg_d = self.graph_aggeration(side_target, side_context, pred_context).permute(0, 2, 1, 3)
        graph_agg_s = self.graph_aggeration2(pred_context, A[:, 1]).permute(0, 2, 1, 3)

        pred_target = pred_target.squeeze(2)
        mask_impute = generate_missing_mask(pred_target.shape, missing_ratio=missing_ratio, device=self.device)

        B, K, L = pred_target.shape
        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):
            current_sample = torch.randn_like(pred_target)
            np_side_info, var, q_dists, p_dists = self.np_model(
                side_context, pred_context, side_target, None, A, context_missing, current_sample.unsqueeze(2)
            )

            for t in range(self.num_steps - 1, -1, -1):
                cond_obs = (mask_impute * pred_target).unsqueeze(1)
                noisy_target = ((1 - mask_impute) * current_sample).unsqueeze(1)
                # stand
                #diff_input = torch.cat([noisy_target, cond_obs, graph_agg_d, graph_agg_s], dim=1)
                diff_input = torch.cat([noisy_target, cond_obs, noisy_target, cond_obs], dim=1)
                predicted = self.diff_model(diff_input, np_side_info, torch.tensor([t]).to(self.device))

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = ((1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]) ** 0.5
                    current_sample += sigma * noise

            imputed_samples[:, i] = current_sample.detach()

        return imputed_samples, mask_impute

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




