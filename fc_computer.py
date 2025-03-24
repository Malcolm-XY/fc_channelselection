# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 23:07:14 2025

@author: 18307
"""

import os
import h5py
import numpy as np
import pandas as pd

from utils import utils_feature_loading
from utils import utils_visualization

def load_global_averages(file_path='Distribution/fc_global_averages.h5'):
    """
    读取 HDF5 文件中的 global_alpha_average, global_beta_average, 和 global_gamma_average 数据。

    Args:
        file_path (str): HDF5 文件的路径，默认是 'Distribution/global_averages.h5'。

    Returns:
        tuple: 包含 global_alpha_average, global_beta_average, 和 global_gamma_average 的元组。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    with h5py.File(file_path, 'r') as f:
        global_alpha_average = f['alpha'][:]
        global_beta_average = f['beta'][:]
        global_gamma_average = f['gamma'][:]
    
    return global_alpha_average, global_beta_average, global_gamma_average

def compute_averaged_fcnetwork(feature, subjects=range(1,16), experiments=range(1,4), draw=True, save=False):
    # 初始化存储结果的列表
    cmdata_averages_dict = []

    # 用于累积频段的所有数据
    all_alpha_values = []
    all_beta_values = []
    all_gamma_values = []

    # 遍历 subject 和 experiment
    for subject in subjects:  # 假设 subjects 是整数
        for experiment in experiments:  # 假设 experiments 是整数
            identifier = f"sub{subject}ex{experiment}"
            print(identifier)
            try:
                # 加载数据
                cmdata_alpha = utils_feature_loading.read_fcs(dataset='seed', identifier=identifier, feature=feature,
                                                              band='alpha')
                cmdata_beta = utils_feature_loading.read_fcs(dataset='seed', identifier=identifier, feature=feature,
                                                             band='beta')
                cmdata_gamma = utils_feature_loading.read_fcs(dataset='seed', identifier=identifier, feature=feature,
                                                              band='gamma')
                # 计算平均值
                cmdata_alpha_averaged = np.mean(cmdata_alpha, axis=0)
                cmdata_beta_averaged = np.mean(cmdata_beta, axis=0)
                cmdata_gamma_averaged = np.mean(cmdata_gamma, axis=0)
                
                # 累积数据
                all_alpha_values.append(cmdata_alpha_averaged)
                all_beta_values.append(cmdata_beta_averaged)
                all_gamma_values.append(cmdata_gamma_averaged)
                
                # # 可视化
                # draw_projection(cmdata_alpha_averaged)
                # draw_projection(cmdata_beta_averaged)
                # draw_projection(cmdata_gamma_averaged)
                
                # 合并同 subject 同 experiment 的数据
                cmdata_averages_dict.append({
                    "subject": subject,
                    "experiment": experiment,
                    "averages": {
                        "alpha": cmdata_alpha_averaged,
                        "beta": cmdata_beta_averaged,
                        "gamma": cmdata_gamma_averaged
                    }
                })
            except Exception as e:
                print(f"Error processing sub {subject} ex {experiment}: {e}")

    # 计算整个数据集的全局平均值
    global_alpha_average = np.mean(all_alpha_values, axis=0)
    global_beta_average = np.mean(all_beta_values, axis=0)
    global_gamma_average = np.mean(all_gamma_values, axis=0)
    global_joint_average = np.mean(np.stack([global_alpha_average, global_beta_average, global_gamma_average], axis=0), axis=0)

    if draw:
        # 输出结果
        utils_visualization.draw_projection(global_alpha_average)
        utils_visualization.draw_projection(global_beta_average)
        utils_visualization.draw_projection(global_gamma_average)
        utils_visualization.draw_projection(global_joint_average)
    
    if save:
        # 检查和创建 Distribution 文件夹
        output_dir = 'Distribution'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # 保存为 HDF5 文件
        file_path = os.path.join(output_dir, 'fc_global_averages.h5')
        with h5py.File(file_path, 'w') as f:
            f.create_dataset('alpha', data=global_alpha_average)
            f.create_dataset('beta', data=global_beta_average)
            f.create_dataset('gamma', data=global_gamma_average)
            f.create_dataset('joint', data=global_joint_average)
        
        print(f"Results saved to {file_path}")
    
    return global_alpha_average, global_beta_average, global_gamma_average, global_joint_average

def compute_distance_matrix(dataset):
    distribution = utils_feature_loading.read_distribution(dataset)
    channel_names, x, y, z = distribution['channel'], distribution['x'], distribution['y'], distribution['z']

    # 将 x, y, z 坐标堆叠成 (N, 3) 形状的矩阵
    coords = np.vstack((x, y, z)).T  # 形状 (N, 3)

    # 计算欧几里得距离矩阵
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    distance_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

    return channel_names, distance_matrix

if __name__ == '__main__':
    # %% Distance Matrix
    # channel_names, distance_matrix = compute_distance_matrix('seed')
    # utils_visualization.draw_projection(distance_matrix)

    # %% PCC
    global_alpha_average, global_beta_average, global_gamma_average, global_joint_average  = compute_averaged_fcnetwork('PCC', subjects=range(1, 16), draw=True, save=True)

    joint = global_joint_average
    joint_mean_1d = np.mean(joint, axis=0)

    # get electrodes
    distribution = utils_feature_loading.read_distribution('seed')
    electrodes = distribution['channel']

    # arrange
    joint_ = pd.DataFrame({'electrodes': electrodes, 'pcc_mean': joint_mean_1d})

    # plot heatmap
    utils_visualization.draw_heatmap_1d(joint_mean_1d, electrodes)

    # get ascending indices
    joint_resorted = joint_.sort_values('pcc_mean', ascending=False)
    utils_visualization.draw_heatmap_1d(joint_resorted['pcc_mean'], joint_resorted['electrodes'])

    # %% PLV
    # global_alpha_average, global_beta_average, global_gamma_average, global_joint_average = compute_averaged_fcnetwork('PLV', subjects=range(1, 16), draw=True, save=True)
    #
    # joint = global_joint_average
    # joint_mean_1d = np.mean(joint, axis=0)
    #
    # # get electrodes
    # distribution = utils_feature_loading.read_distribution('seed')
    # electrodes = distribution['channel']
    #
    # # arrange
    # joint_ = pd.DataFrame({'electrodes': electrodes, 'pcc_mean': joint_mean_1d})
    #
    # # plot heatmap
    # utils_visualization.draw_heatmap_1d(joint_mean_1d, electrodes)
    #
    # # get ascending indices
    # joint_resorted = joint_.sort_values('pcc_mean', ascending=False)
    # utils_visualization.draw_heatmap_1d(joint_resorted['pcc_mean'], joint_resorted['electrodes'])