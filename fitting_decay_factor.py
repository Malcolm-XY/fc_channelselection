# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 14:10:03 2025

@author: usouu
"""
import numpy as np
import pandas as pd

import fc_computer
import channel_selection_weight_mapping

from utils import utils_visualization

# %% ====== STEP 1: 准备目标向量 r_target 和输入矩阵 ======
# %% Label Driven Ranking; Fitting target
weight_mean, index = channel_selection_weight_mapping.draw_weight_mapping(transformation='log', ranking_method='label_driven_mi')

weight_mean = fc_computer.normalize_matrix(weight_mean)

df_label_driven_ranking = pd.DataFrame({'index': index,'weight_mean': weight_mean})

df_label_driven_ranking = df_label_driven_ranking.sort_values(by='index').reset_index(drop=True)

r_target = df_label_driven_ranking['weight_mean']
r_target = fc_computer.normalize_matrix(r_target)
r_target = np.sqrt(r_target)

# %% 
channel_names, distance_matrix = fc_computer.compute_distance_matrix('seed', method='stereo')
distance_matrix = fc_computer.normalize_matrix(distance_matrix)

# utils_visualization.draw_projection(distance_matrix)

factor_matrix = fc_computer.compute_volume_conduction_factors(distance_matrix, method='exponential', params={'sigma': 2.7})
factor_matrix = fc_computer.normalize_matrix(factor_matrix)
# utils_visualization.draw_projection(factor_matrix)

_, _, _, global_joint_average = fc_computer.load_global_averages(feature='PCC')
connectivity_matrix = fc_computer.normalize_matrix(global_joint_average)
# utils_visualization.draw_projection(connectivity_matrix)

differ_PCC_DM = connectivity_matrix - factor_matrix
differ_PCC_DM = fc_computer.normalize_matrix(differ_PCC_DM)
# utils_visualization.draw_projection(differ_PCC_DM)

r_fitting = np.mean(differ_PCC_DM, axis=0)
r_fitting = fc_computer.normalize_matrix(r_fitting)
r_fitting = np.sqrt(r_fitting)

# %% Fitting; Solve params and compute_volume_conduction_factors
# ====== STEP 2: 定义损失函数，用于最小化 r_fitting 与 r_target 的差异 ======
from scipy.optimize import minimize
def loss_fn(params):
    sigma = params[0]
    if sigma <= 0:
        return np.inf  # 防止负值或零
    
    # 计算 factor_matrix
    factor_matrix = fc_computer.compute_volume_conduction_factors(
        distance_matrix,
        method='exponential',
        params={'sigma': sigma}
    )
    factor_matrix = fc_computer.normalize_matrix(factor_matrix)

    # 差异矩阵（即已减去体积传导影响的连接图）
    differ_PCC_DM = connectivity_matrix - factor_matrix
    differ_PCC_DM = fc_computer.normalize_matrix(differ_PCC_DM)

    # 每个通道的平均连接强度作为拟合向量
    r_fitting = np.mean(differ_PCC_DM, axis=0)

    # 返回 MSE 损失
    return np.mean((r_fitting - r_target) ** 2)

# ====== STEP 3: 启动拟合优化 ======
result = minimize(loss_fn, x0=[2.0], bounds=[(0.1, 20.0)])
print("最佳 sigma:", result.x[0])
print("最小 MSE:", result.fun)