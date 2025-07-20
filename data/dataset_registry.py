# -*- coding: utf-8 -*-
"""
数据集注册中心，用于集中管理所有数据集的配置信息
"""

# 数据集配置字典
DATASET_CONFIG = {
    "BJAir": {
        "location_path": "data/bjair/stations.txt",
        "data_path": "data/bjair/processed_raw.txt",
        "meta_path": "data/bjair/meta_data.pkl",
        "test_nodes_path": "data/bjair/test_nodes.npy",
        "all_attrs": ["PM25_Concentration", "PM10_Concentration", "NO2_Concentration",
                      "CO_Concentration", "O3_Concentration", "SO2_Concentration"],
        "missing_attrs": ["PM25_Missing", "PM10_Missing", "NO2_Missing",
                          "CO_Missing", "O3_Missing", "SO2_Missing"],
        "load_loc_method": "load_loc_distance"
    },
    "BJAir2017": {
        "location_path": "data/bjair2017/stations.csv",
        "data_path": "data/bjair2017/raw_data.csv",
        "meta_path": "data/bjair2017/meta_data.pkl",
        "test_nodes_path": "data/bjair2017/test_nodes.npy",
        "all_attrs": ["PM25_Concentration", "PM10_Concentration", "NO2_Concentration"],
        "missing_attrs": ["PM25_Missing", "PM10_Missing", "NO2_Missing"],
        "load_loc_method": "load_loc_distance"
    },
    "IntelLab": {
        "location_path": "data/intellab/adj.txt",
        "data_path": "data/intellab/raw_data.csv",
        "meta_path": "data/intellab/meta_data.pkl",
        "test_nodes_path": "data/intellab/test_nodes.npy",
        "all_attrs": ["light"],
        "missing_attrs": ["temperature_Missing", "humidity_Missing", "light_Missing", "voltage_Missing"],
        "load_loc_method": "load_loc_direct"
    },
    "Covid19": {
        "location_path": "data/covid19/stations.csv",
        "data_path": "data/covid19/final.csv",
        "meta_path": "data/covid19/meta_data.pkl",
        "test_nodes_path": "data/covid19/test_nodes.npy",
        "all_attrs": [],  # 动态设置
        "missing_attrs": [],  # 动态设置
        "load_loc_method": "load_loc_distance"
    },
    "London": {
        "location_path": "data/london/stations.csv",
        "data_path": "data/london/raw_data.csv",
        "meta_path": "data/london/meta_data.pkl",
        "test_nodes_path": "data/london/test_nodes.npy",
        "all_attrs": ["PM25_Concentration", "PM10_Concentration", "NO2_Concentration"],
        "missing_attrs": ["PM25_Missing", "PM10_Missing", "NO2_Missing"],
        "load_loc_method": "load_loc_distance"
    },
    "Metra": {
        "data_path": "data/metr/node_values.npy",
        "test_nodes_path": "data/metr/test_nodes.npy",
        "adj_path": "data/metr/adj_mat.npy",
        "load_loc_method": "load_adj_direct"
    },
    "PEMS03": {
        "data_path": "data/PEMS03/data.npz",
        "test_nodes_path": "data/PEMS03/test_nodes.npy",
        "adj_path": "data/PEMS03/adj_pems03.pkl",
        "load_loc_method": "load_adj_pickle"
    },
    "PEMS03_small": {
        "data_path": "data/PEMS03/data_small.npz",
        "test_nodes_path": "data/PEMS03/test_nodes.npy",
        "adj_path": "data/PEMS03/adj_pems03_small.pkl",
        "load_loc_method": "load_adj_pickle"
    }
}

def get_dataset_config(dataset_name):
    """获取指定数据集的配置"""
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"未知的数据集: {dataset_name}")
    return DATASET_CONFIG[dataset_name]
