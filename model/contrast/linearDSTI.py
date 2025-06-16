
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.inner.st_module.net_module import Conv1d_with_init, MultiConv1d
from model.inner.st_module.graph_layer import GraphAggregation
from model.inner.st_module.time_layer import TemporalLearning, TConv1d


class linearDSTI(nn.Module):
    def __init__(self, config):
        super().__init__()
        # config['covariate_dim'] = 14
        tcn_channels = [config['emd_channel']] + config['tcn_channels']

        # 特征嵌入层
        self.feature_embedding = Conv1d_with_init(config['input_dim'], config['emd_channel'])
        # 时空联合学习模块
        self.spatio_temporal = nn.ModuleList([
            nn.ModuleDict({
                'graph_agg': GraphAggregation(tcn_channels[i], tcn_channels[i], order=2),
                'temp_conv': TemporalLearning(tcn_channels[i], tcn_channels[i + 1],
                                              config['tcn_kernel_size'], 2 ** i, config['dropout']),
                'residual': Conv1d_with_init(tcn_channels[i], tcn_channels[i + 1], 1)
            }) for i in range(len(tcn_channels) - 1)
        ])

        # 协变量处理
        if config['covariate_dim'] > 0:
            self.side_encoder = nn.ModuleList([
                Conv1d_with_init(config['covariate_dim'], config['tcn_channels'][i])
                for i in range(len(config['tcn_channels']))
            ])

        # 整合后的线性观测模型
        self.observation = MultiConv1d(
            in_channels=sum(config["tcn_channels"]),
            out_channels=config['input_dim'],
            num_layers=3,
            channel_reduction='half',
            dropout=0.1
        )
        # 空值时的 pred_target, q(Y|X, C)
        self.empty_token = nn.Parameter(torch.zeros([1, 1, 1, 1]))

    def forward(self, x_context, y_context, x_target, y_target, adj, missing_mask):
        # 输入预处理
        b, num_n, dy, t = y_context.shape
        num_m = adj.shape[2]

        y_target = self.empty_token.repeat([b, num_m, 1, t])
        y_target = self.feature_embedding(y_target)
        y_context = self.feature_embedding(y_context)

        # 时空特征传播
        y = []
        for i, layer in enumerate(self.spatio_temporal):
            # 图聚合
            target_clone = y_target.clone()
            context_clone = y_context.clone()

            y_target = layer['graph_agg'](y_context, y_target, adj, missing_mask)

            # 时间卷积
            y_target = layer['temp_conv'](y_target)
            y_context = layer['temp_conv'](y_context)

            # 协变量融合
            if hasattr(self, 'side_encoder'):
                y_target += self.side_encoder[i](x_target)
                y_context += self.side_encoder[i](x_context)

            # 残差连接
            if i > 0:
                y_target = F.relu(y_target + layer['residual'](target_clone))
                y_context = F.relu(y_context + layer['residual'](context_clone))
            y.append(y_target)
        y = torch.cat(y, dim=2)
        # 最终预测
        return self.observation(y)