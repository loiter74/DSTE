# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 22:32:20 2023

@author: LUN076
"""
import torch
import torch.nn as nn

from model.inner.st_module.time_layer import MultiTConv1d
from model.inner.st_module.conv_layer import Conv1d_with_init

class DynamicAgg(nn.Module):
    def __init__(self, pred_in=1, feat_in=7, channels=64, out=64 ,dropout=0.1):
        super(DynamicAgg, self).__init__()
        self.proj_feat = MultiTConv1d(in_channels=feat_in, hidden_channels=channels, out_channels=channels, num_layers=4)
        self.proj_pred = MultiTConv1d(in_channels=pred_in, hidden_channels=channels, out_channels=channels, num_layers=4)
        self.proj_out = MultiTConv1d(in_channels=channels, hidden_channels=channels, out_channels=out, num_layers=4)
        self.norm1 = nn.LayerNorm(channels)

        self.relu = nn.ReLU()
    def forward(self, feat_target, feat_context, pred_context):
        '''
        new_adj: b, dx, m, n
        pred_context: b, n, dy, t
        '''
        feat_context = self.proj_feat(feat_context)
        feat_target = self.proj_feat(feat_target)
        
        new_adj = torch.einsum("bmct,bnct->bcmn", feat_target, feat_context)
        pred_context = self.proj_pred(pred_context)
        pred_target = torch.einsum("bcmn,bnct->bmct", new_adj, pred_context)
        #pred_target = self.relu(pred_target)
        pred_target = self.norm1(pred_target.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        pred_target = self.proj_out(pred_target)

        return pred_target

class StaticAgg(nn.Module):
    def __init__(self, pred_in=1, channels=128, out_channels=128, dropout=0.1):
        super(StaticAgg, self).__init__()
        
        self.proj_pred = Conv1d_with_init(in_channels=pred_in, out_channels=channels)
        self.proj_out = Conv1d_with_init(in_channels=channels, out_channels=out_channels)
        self.norm1 = nn.LayerNorm(channels)
    def forward(self, pred_context, adj):
        '''
        new_adj: b, dx, m, n
        pred_context: b, n, dy, t
        '''
        pred_context = self.proj_pred(pred_context)  #[20, 3, 20]
        pred_target = torch.einsum("bmn,bnct->bmct", adj, pred_context)
        pred_target = self.norm1(pred_target.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        pred_target = self.proj_out(pred_target)
        return pred_target

    
class GraphAggregation(nn.Module):
    def __init__(self, c_in, c_out, order, dropout=0.1):
        super().__init__()
        c_in = (order + 1) * c_in  # 1 for residual connection
        self.mlp = Conv1d_with_init(c_in, c_out, 1, dropout=dropout, activ="relu")
        self.order = order

    def forward(self, feat_context, feat_target, adj, missing_mask_context):
        """
        Cross set graph neural network
        m: target set
        n: context set
        Args:
            feat_context: [batch, num_n, d, time]
            feat_target: [batch, num_m, d, time]
            adj: adjacency matrix for target_node [batch, k_hop, num_m, num_n]
            missing_mask_context: index the missing nodes (1: missing) [batch, time, num_n]
        Returns:
            feat_target: [batch, num_m, d_o, time]
        """
        out = [feat_target]
        for i in range(self.order):
            out += [self.aggregate(feat_context, feat_target, adj[:, i], missing_mask_context)]
        out = torch.cat(out, dim=2)
        feat_target = self.mlp(out)
        return feat_target

    def aggregate(self, feat_context, feat_target, adj, missing_index_context):
        feat_context, feat_target = feat_context.permute(0, 3, 1, 2), feat_target.permute(0, 3, 1,
                                                                                          2)  # [batch, time, num_n, d]
        b, t, num_n = feat_context.shape[:3]
        num_m = feat_target.shape[2]
        adj = torch.tile(adj.unsqueeze(1), [1, t, 1, 1])
        missing_index_context = torch.tile(missing_index_context.unsqueeze(2), [1, 1, num_m, 1])
        adj = adj * (1 - missing_index_context)
        norm_adj = adj / (torch.sum(adj, dim=-1, keepdim=True) + 1)  # 1 for the target node
        feat_target = (feat_target + torch.sum(feat_context.unsqueeze(2) * norm_adj.unsqueeze(-1), dim=-2))
        feat_target = feat_target.permute(0, 2, 3, 1)
        return feat_target

    
    
    
    
    
    
    
    
    