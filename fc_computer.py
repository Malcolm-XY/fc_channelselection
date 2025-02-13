# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 23:07:14 2025

@author: 18307
"""

import os
import h5py
import numpy as np

from utils import load_cmdata2d
from utils import draw_projection

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

def get_averaged_fcnetwork(feature, subjects=range(1,16), experiments=range(1,4), draw=True, save=False):
    # 初始化存储结果的列表
    cmdata_averages_dict = []

    # 用于累积频段的所有数据
    all_alpha_values = []
    all_beta_values = []
    all_gamma_values = []

    # 遍历 subject 和 experiment
    for subject in subjects:  # 假设 subjects 是整数
        for experiment in experiments:  # 假设 experiments 是整数
            print(f'sub: {subject} ex: {experiment}')
            try:
                # 加载数据
                cmdata_alpha = load_cmdata2d(feature, 'alpha', f'sub{subject}ex{experiment}')
                cmdata_beta = load_cmdata2d(feature, 'beta', f'sub{subject}ex{experiment}')
                cmdata_gamma = load_cmdata2d(feature, 'gamma', f'sub{subject}ex{experiment}')
                
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
    
    if draw:
        # 输出结果
        draw_projection(global_alpha_average)
        draw_projection(global_beta_average)
        draw_projection(global_gamma_average)
    
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
        
        print(f"Results saved to {file_path}")
    
    return global_alpha_average, global_beta_average, global_gamma_average

def compute_corr_matrices(eeg_data, samplingrate, window=1, overlap=0, verbose=True, visualization=True):
    """
    Compute correlation matrices for EEG data using a sliding window approach.
    
    Parameters:
        eeg_data (numpy.ndarray): EEG data with shape (channels, time_samples).
        samplingrate (int): Sampling rate of the EEG data in Hz.
        window (float): Window size in seconds for segmenting EEG data.
        overlap (float): Overlap fraction between consecutive windows (0 to 1).
        verbose (bool): If True, prints progress.
        visualization (bool): If True, displays correlation matrices.
    
    Returns:
        list of numpy.ndarray: List of correlation matrices for each window.
    """
    # Compute step size based on overlap
    step = int(samplingrate * window * (1 - overlap))  # Step size for moving window
    segment_length = int(samplingrate * window)

    # Split EEG data into overlapping windows
    split_segments = [
        eeg_data[:, i:i + segment_length] 
        for i in range(0, eeg_data.shape[1] - segment_length + 1, step)
    ]

    # Compute correlation matrices
    corr_matrices = []
    for idx, segment in enumerate(split_segments):
        if segment.shape[1] < segment_length:
            continue  # Skip incomplete segments
        
        # Compute Pearson correlation
        corr_matrix = np.corrcoef(segment)
        corr_matrices.append(corr_matrix)

        if verbose:
            print(f"Computed correlation matrix {idx + 1}/{len(split_segments)}")

    # Optional: Visualization of correlation matrices
    if visualization and corr_matrices:
        avg_corr_matrix = np.mean(corr_matrices, axis=0)
        draw_projection(avg_corr_matrix)

    return corr_matrices

if __name__ == '__main__':
    get_averaged_fcnetwork('PCC', save=True)
    # get_averaged_fcnetwork('PLV', save=True)