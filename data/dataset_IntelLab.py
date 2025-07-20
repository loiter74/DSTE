# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 15:11:26 2024

@author: dell
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

from .dataset_base import BaseDataset
from torch.utils.data import DataLoader, Dataset

class IntelLab(BaseDataset):
    def __init__(self, opt, mode='train'):
        super().__init__(opt)
   
        location_path = 'data/intellab/adj.txt'
        data_path = 'data/intellab/raw_data.csv'
        meta_path = 'data/intellab/meta_data.pkl'
        test_nodes_path = 'data/intellab/test_nodes.npy'
        
        # location_path = 'intellab/adj.txt'
        # data_path = 'intellab/raw_data.csv'
        # meta_path = 'intellab/meta_data.pkl'
        # test_nodes_path = 'intellab/test_nodes.npy'
          
        self.pred_attrs = [opt.pred_attr]
        self.drop_attrs = ['light']
        self.drop_attrs.remove(opt.pred_attr)
        self.drop_attrs += ['temperature_Missing','humidity_Missing','light_Missing', 'voltage_Missing']
        
        self.A = self.load_loc(location_path)
        # print(self.A.shape)
        self.raw_data, norm_info = self.load_feature(data_path, meta_path, self.time_division[mode])
        
        # get data division index
        self.test_node_index = self.get_node_division(test_nodes_path, num_nodes=self.raw_data['pred'].shape[0])
        self.train_node_index = np.setdiff1d(np.arange(self.raw_data['pred'].shape[0]), self.test_node_index)
        # add norm info
        self.add_norm_info(norm_info.at['mean', opt.pred_attr], norm_info.at['scale', opt.pred_attr])
        # print(norm_info.at['mean', opt.pred_attr], norm_info.at['scale', opt.pred_attr]) # 84.6456151224047 81.20440084096904
        # print(self.test_node_index.shape)
        # print(self.train_node_index.shape) # 10 25
        
    def load_loc(self, loaction_path):
        # 读取邻接矩阵文件
        with open(loaction_path, 'r') as f:
            lines = f.readlines()
        
        # 提取源节点、目标节点和权重
        adj = np.zeros((58, 58))
        edges = []
        weights = []
        for line in lines:
            items = line.split()
            # print(items)
            source = int(items[0])
            target = int(items[1])
            weight = float(items[2])
            edges.append((source, target))
            weights.append(weight)
            adj[source][target] = weight
            
        # 创建双向矩阵
        num_nodes = 58
        bi_adj_matrix = np.zeros((num_nodes, num_nodes, 2))
        
        # 填充双向矩阵
        for i, (source, target) in enumerate(edges):
            bi_adj_matrix[source, target, 0] = weights[i]  # 正向权重
            bi_adj_matrix[target, source, 1] = weights[i]  # 反向权重
            
        return adj
        # return np.ones((58, 58))
        
    def load_feature(self, data_path, meta_path, time_division, delete_col=None):
        multimodal = pd.read_csv(data_path, header=0)
        # print(beijing_multimodal.shape)
        # get normalization info
        # sorry we also involve val, test data in normalization, due to the coding complexity
        # print('Computing normalization info...')
        with open(meta_path, 'rb') as f:
            cont_cols = pickle.load(f)['cont_cols']  # list
            
            # print(cont_cols)
            # print(len(cont_cols))
        feat_scaler = StandardScaler()
        # print(cont_cols)
        multimodal[cont_cols] = feat_scaler.fit_transform(multimodal[cont_cols])
        norm_info = pd.DataFrame([feat_scaler.mean_, feat_scaler.scale_, feat_scaler.var_], columns=cont_cols, index=['mean', 'scale', 'var'])

        # print('Loading air quality features...')
        data = {'feat': [],
                'pred': [],
                'missing': [],
                'time': []}
        # print(self.pred_attrs, "\n",self.drop_attrs)
        for id, station_aq in multimodal.groupby('moteid'):
            station_aq = station_aq.set_index("time").drop(columns=['moteid'])
            if delete_col is not None:
                station_aq = station_aq.drop(columns=delete_col)
            # split data into features and labels
          
            data['feat'].append(station_aq.drop(columns=self.pred_attrs+self.drop_attrs).to_numpy()[np.newaxis])
            data['missing'].append(station_aq[[attr.split('_')[0]+'_Missing' for attr in self.pred_attrs]].to_numpy()[np.newaxis])
            data['pred'].append(station_aq[self.pred_attrs].to_numpy()[np.newaxis])
            # data['time'].append()
        
        # for i in range(len(data['feat'])):
        #     print(data['feat'][i].shape)

        data_length = data['feat'][0].shape[1]
        start_index, end_index = int(time_division[0] * data_length), int(time_division[1] * data_length)
        data['feat'] = np.concatenate(data['feat'], axis=0)[:, start_index:end_index, :]
        data['missing'] = np.concatenate(data['missing'], axis=0)[:, start_index:end_index, :]
        data['pred'] = np.concatenate(data['pred'], axis=0)[:, start_index:end_index, :]
        # data['time'] = station_aq[start_index:end_index].index.values.astype(np.datetime64)
        # print(station_aq[start_index:end_index].index)s
        data['time'] = pd.to_datetime(station_aq[start_index:end_index].index.values)
        data['time'] = data['time'].values
        data['time'] = ((data['time'] - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's'))
        

        return data, norm_info    
    
def get_dataloader(opt):
    dataset = IntelLab(opt, mode="train")
    train_loader = DataLoader(
        dataset, batch_size=opt.batch_size, num_workers=4, shuffle=True
        )
    dataset_valid =  IntelLab(opt, mode="val")
    valid_loader = DataLoader(
        dataset_valid, batch_size=opt.batch_size, num_workers=4, shuffle=False
    )
    dataset_test =  IntelLab(opt, mode="test")
    test_loader = DataLoader(
        dataset_test, batch_size=opt.batch_size, num_workers=4, shuffle=False
    )
    return train_loader, valid_loader, test_loader

if __name__ == "__main__":
    
    import argparse
    import torch
    
    parser = argparse.ArgumentParser(description="DSTE")
    parser.add_argument('--pred_attr', type=str, default='temperature', help='Which AQ attribute to infer')
    parser.add_argument('--t_len', type=int, default=24, help='time window for inference')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
    parser.add_argument('--num_train_target', type=int, default=3, help='# of target set for training at the start')
    
    
    opt = parser.parse_args()
    intellab = IntelLab(opt)
    dataloader = torch.utils.data.DataLoader(intellab, batch_size=opt.batch_size, shuffle=False, num_workers=1)
    
    
    for data in dataloader:  # inner loop within one epoch
        # print(data.keys())
        for key in data.keys():
            print(key, data[key].shape)
        break

    