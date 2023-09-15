# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 17:30:53 2023

@author: dell
"""
import pandas as pd

df_gt = pd.read_csv(
    "pm25_missing.txt",
    index_col="datetime",
    parse_dates=True,
)
df = pd.read_csv(
    "pm25_ground.txt",
    index_col="datetime",
    parse_dates=True,
)


shuffled_df_gt = df_gt.sample(frac=1, axis=1, random_state=42)

shuffled_df = df_gt.sample(frac=1, axis=1, random_state=42)

shuffled_df_gt.to_csv('pm25_missing_shuffled.txt') 
shuffled_df.to_csv('pm25_ground_shuffled.txt') 
