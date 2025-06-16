import torch
import torch.nn as nn

from model.contrast import DiffusionEmbedding
from model.inner.st_module.net_module import Conv1d_with_init
from model.inner.st_module.graph_layer import GraphAggregation
from model.inner.st_module.time_layer import TemporalLearning, TConv1d, LSTM


class DenoiseNetwork(nn.Module):
    def __init__(self, input_dim, covariant_dim, emd_channel, tcn_channels, tcn_kernel_size, dropout, ):
        """
        Args:
            tcn_channels: list
            tcn_kernel_size: int
            dropout: float
        """
        super().__init__()
        self.channels = tcn_channels
        side_channels = [covariant_dim] + tcn_channels # 循环初始化输入
        tcn_channels = [emd_channel] + tcn_channels

        self.feature_embedding = Conv1d_with_init(input_dim, emd_channel)
        # self.aggr_encoding = nn.ModuleList([nn.Conv1d(tcn_channels[i], tcn_channels[i+1], 1) for i in range(len(tcn_channels)-1)])
        # todo: hard coding part, need to be changed
        self.graph_aggregator = nn.ModuleList(
            [GraphAggregation(tcn_channels[i], tcn_channels[i], order=2) for i in range(len(tcn_channels) - 1)])
        self.time_learning = nn.ModuleList(
            [TemporalLearning(tcn_channels[i], tcn_channels[i + 1], tcn_kernel_size, 2 ** i, dropout) for i in
             range(len(tcn_channels) - 1)])
        self.time_learning2 = nn.ModuleList(
            [TConv1d(tcn_channels[i], tcn_channels[i + 1], tcn_kernel_size, 2 ** i, dropout) for i in
             range(len(tcn_channels) - 1)])
        # lstm不适合非序列训练
        self.time_learning3 = nn.ModuleList([LSTM(tcn_channels[i], tcn_channels[i + 1], dropout) for i in
             range(len(tcn_channels) - 1)])

        if covariant_dim > 0:
            self.side_encoding = nn.ModuleList(
                [Conv1d_with_init(side_channels[i], side_channels[i + 1], 1) for i in
                 range(len(side_channels) - 1)])
        # self.output_projection = nn.ModuleList([Conv1d(tcn_channels[i], tcn_channels[i], 1, dropout=dropout) for i in range(1, len(tcn_channels))]
        # residual connection
        self.residual = nn.ModuleList(
            [Conv1d_with_init(tcn_channels[i], tcn_channels[i + 1], 1) for i in range(len(tcn_channels) - 1)])
        # self.empty_token = nn.Parameter(torch.zeros([1, 1, emd_channel, 1]))
        self.dropout = nn.Dropout(dropout)

        # 空值时的 pred_target, q(Y|X, C)
        self.empty_token = nn.Parameter(torch.zeros([1, 1, emd_channel, 1]))
        self.diff_time_learning = nn.ModuleList([DiffusionEmbedding(num_steps=100, embedding_dim=tcn_channels[i]) for i in range(len(tcn_channels) - 1)])
        self.group_norm_layers = nn.ModuleList([nn.GroupNorm(4, tcn_channels[i]) for i in range(1, len(tcn_channels))])

    def forward(self, x_context, y_context, x_target, y_target, adj, missing_mask_context, diffusion_step):
        target, context = y_target, y_context
        b, num_n, dy, t = y_context.shape
        num_m = adj.shape[2]

        if target is None:
            target = self.empty_token.repeat([b, num_m, 1, t])
        else:
            target = self.feature_embedding(target)

        context = self.feature_embedding(context)

        d_t = []
        d_c = []
        for i in range(len(self.channels) + 1 - 1):
            target = target + self.diff_time_learning[i](diffusion_step).unsqueeze(1).unsqueeze(-1)
            target_resi = target.clone()
            context_resi = context.clone()
            #context = self.aggr_encoding[i](context.view([b*num_n, -1, t])).view([b, num_n, -1, t])
            target = self.graph_aggregator[i](context, target, adj, missing_mask_context)
            # context, target = torch.relu(self.dropout(context)), torch.relu(self.dropout(target))
            # temporal convolution    attn + lstm
            target = self.time_learning[i](target)
            context = self.time_learning[i](context) #使用TCN学习减少复杂度

            # if x_target is not None: # side information
            #     x_target = self.side_encoding[i](x_target)
            #     target = target + x_target
            #     x_context = self.side_encoding[i](x_context)
            #     context = context + x_context
            #
            if i > 0:  # residual connection
                target = (target + self.residual[i](target_resi)).permute(0, 2, 1, 3)
                context = (context + self.residual[i](context_resi)).permute(0, 2, 1, 3)
                target = self.group_norm_layers[i](target).permute(0, 2, 1, 3)
                context = self.group_norm_layers[i](context).permute(0, 2, 1, 3)


            d_t += [target]
            d_c += [context]
        return d_c, d_t


