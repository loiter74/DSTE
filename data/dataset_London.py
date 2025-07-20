# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 15:13:49 2023

@author: dell
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from math import radians, cos, sin, asin, sqrt
from .dataset_base import BaseDataset
# from dataset_base import BaseDataset

from torch.utils.data import DataLoader, Dataset

class LondonAir(BaseDataset):
    def __init__(self, opt, mode='train'):
        super().__init__(opt, mode)
   
        location_path = 'data/london/stations.csv'
        data_path = 'data/london/raw_data.csv'
        meta_path = 'data/london/meta_data.pkl'
        test_nodes_path = 'data/london/test_nodes.npy'
        
        # location_path = 'london/stations.csv'
        # data_path = 'london/raw_data.csv'
        # meta_path = 'london/meta_data.pkl'
        # test_nodes_path = 'london/test_nodes.npy'
        
        self.pred_attrs = [opt.pred_attr]
        self.drop_attrs = ['PM25_Concentration','PM10_Concentration','NO2_Concentration']
        self.drop_attrs.remove(opt.pred_attr)
        self.drop_attrs += ['PM25_Missing','PM10_Missing', 'NO2_Missing']
        
        
        
        self.A = self.load_loc(location_path)
        self.raw_data, norm_info = self.load_feature(data_path, meta_path, self.time_division[mode])
        
        # get data division index
        self.test_node_index = self.get_node_division(test_nodes_path, num_nodes=self.raw_data['pred'].shape[0])
        self.train_node_index = np.setdiff1d(np.arange(self.raw_data['pred'].shape[0]), self.test_node_index)
        # add norm info
        self.add_norm_info(norm_info.at['mean', opt.pred_attr], norm_info.at['scale', opt.pred_attr], norm_info.at["var", opt.pred_attr])
        # print(norm_info.at['mean', opt.pred_attr], norm_info.at['scale', opt.pred_attr]) # 84.6456151224047 81.20440084096904
        # print(self.test_node_index.shape)
        # print(self.train_node_index.shape) # 10 25
    def load_loc(self, aq_location_path, build_adj=True):
        """
        Args:
            build_adj: if True, build adjacency matrix else return horizontal and vertical distance matrix
        Returns:

        """
        # print('Loading station locations...')
        # load air quality station locations data
        beijing_location = pd.read_csv(aq_location_path)

        # load station locations for adj construction
        beijing_location = beijing_location.sort_values(by=['station_id'])
        num_station = len(beijing_location)

        if build_adj:
            # build adjacency matrix for each target node
            A = np.zeros((num_station, num_station))
            for t in range(num_station):
                for c in range(num_station):
                    dis = self.haversine(beijing_location.at[t, 'longitude'],
                                                beijing_location.at[t, 'latitude'],
                                                beijing_location.at[c, 'longitude'],
                                                beijing_location.at[c, 'latitude'])
                    A[t, c] = dis
            # Gaussian and normalization
            A = np.exp(- 0.5 * (A / np.std(A)) ** 2)
        else:
            A = np.zeros((num_station, num_station, 2))
            for t in range(num_station):
                for c in range(num_station):
                    A[t, c, 0] = beijing_location.at[t, 'longitude'] - beijing_location.at[c, 'longitude']
                    A[t, c, 1] = beijing_location.at[t, 'latitude'] - beijing_location.at[c, 'latitude']
        return A
    
    def load_feature(self, data_path, meta_path, time_division, delete_col=None):
        beijing_multimodal = pd.read_csv(data_path, header=0)
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
        beijing_multimodal[cont_cols] = feat_scaler.fit_transform(beijing_multimodal[cont_cols])
        norm_info = pd.DataFrame([feat_scaler.mean_, feat_scaler.scale_, feat_scaler.var_], columns=cont_cols, index=['mean', 'scale', 'var'])

        # print('Loading air quality features...')
        data = {'feat': [],
                'pred': [],
                'missing': [],
                'time': []}
        # print(self.pred_attrs, "\n",self.drop_attrs)
        for id, station_aq in beijing_multimodal.groupby('station_id'):
            station_aq = station_aq.set_index("time").drop(columns=['station_id'])
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
    dataset = LondonAir(opt, mode="train")
    train_loader = DataLoader(
        dataset, batch_size=opt.batch_size, num_workers=4, shuffle=True
        )
    dataset_valid =  LondonAir(opt, mode="val")
    valid_loader = DataLoader(
        dataset_valid, batch_size=opt.batch_size, num_workers=4, shuffle=False
    )
    dataset_test =  LondonAir(opt, mode="test")
    test_loader = DataLoader(
        dataset_test, batch_size=opt.batch_size, num_workers=4, shuffle=False
    )
    return train_loader, valid_loader, test_loader

if __name__ == "__main__":
    
    import argparse
    import torch
    
    parser = argparse.ArgumentParser(description="DSTE")
    parser.add_argument('--pred_attr', type=str, default='PM25_Concentration', help='Which AQ attribute to infer')
    parser.add_argument('--t_len', type=int, default=24, help='time window for inference')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
    parser.add_argument('--num_train_target', type=int, default=3, help='# of target set for training at the start')
    
    
    opt = parser.parse_args()
    londonair = LondonAir(opt)
    dataloader = torch.utils.data.DataLoader(londonair, batch_size=opt.batch_size, shuffle=False, num_workers=1)
    
    
    for data in dataloader:  # inner loop within one epoch
        # print(data.keys())
        for key in data.keys():
            print(key, data[key].shape)
        break

    
    
    
    