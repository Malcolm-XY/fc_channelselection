# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:43:47 2025

@author: 18307
"""
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from utils import utils_feature_loading

def get_ranking_weight(ranking='label_driven_mi'):
    # define path
    path_current = os.getcwd()
    
    path_ranking = os.path.join(path_current, 'Distribution', 'electrodes_ranking.xlsx')
    
    # read xlxs; electrodes ranking
    ranking = pd.read_excel(path_ranking, sheet_name=ranking, engine='openpyxl')
    ranking = ranking['mean']
    
    return ranking    

def get_ranking(ranking='label_driven_mi'):
    # define path
    path_current = os.getcwd()
    
    path_ranking = os.path.join(path_current, 'Distribution', 'electrodes_ranking.xlsx')
    
    # read xlxs; electrodes ranking
    ranking = pd.read_excel(path_ranking, sheet_name=ranking, engine='openpyxl')
    ranking = ranking['index(in origin dataset)']
    
    return ranking    

def draw_weight_mapping(ranking_method='label_driven_mi', offset=0, transformation='log', reverse=False):
    # 获取数据
    ranking = get_ranking(ranking_method)
    weight_mean = get_ranking_weight(ranking_method)  # 假设它返回一个与 electrodes 对应的值列表
    if reverse:
        weight_mean = 1 - weight_mean
    distribution = utils_feature_loading.read_distribution('seed')
    
    dis_t = distribution.iloc[ranking]
    
    x = np.array(dis_t['x'])
    y = np.array(dis_t['y'])
    electrodes = dis_t['channel']
    
    # 归一化 label_driven_mi_mean 以适应颜色显示（假设它是数值列表）
    if transformation == 'log':
        values = np.array(np.log(weight_mean) + offset)
    else: 
        values = np.array(weight_mean + offset)
    
    # 绘制散点图
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(x, y, c=values, cmap='coolwarm', s=100, edgecolors='k')
    
    # 添加颜色条
    cbar = plt.colorbar(sc)
    cbar.set_label('Label Driven MI Mean')
    
    # 标注电极通道名称
    for i, txt in enumerate(electrodes):
        plt.text(x[i], y[i], txt, fontsize=9, ha='right', va='bottom')
    
    # 设置标题和坐标轴
    plt.title("Label Driven MI Mean Distribution on Electrodes")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plt.show()
    
    return weight_mean

if __name__ == '__main__':    
    draw_weight_mapping(transformation='log', ranking_method='label_driven_mi')
    # draw_weight_mapping(transformation=None, ranking_method='data_driven_mi')
    # draw_weight_mapping(transformation=None, ranking_method='data_driven_pcc')
    # draw_weight_mapping(transformation=None, ranking_method='data_driven_plv')

    # draw_weight_mapping(transformation=None, ranking_method='data_driven_pcc_dm', reverse=False)    
    # draw_weight_mapping(transformation=None, ranking_method='data_driven_pcc_dm', reverse=True)