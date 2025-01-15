# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 21:18:58 2024

@author: 18307
"""
import os
import pandas
import numpy
import h5py
import scipy

def get_label():
    # path
    path_current = os.getcwd()
    path_parent = os.path.dirname(path_current)
    
    path_labels = os.path.join(path_parent, 'data', 'SEED', 'functional connectivity', 'labels.txt')
    
    # read txt; original channel distribution
    labels = pandas.read_csv(path_labels, sep='\t', header=None).to_numpy().flatten()
    
    print('Labels Reading Done')
    
    return labels

def raed_labels(path_txt):
    # read txt; original channel distribution
    labels = pandas.read_csv(path_txt, sep='\t', header=None).to_numpy().flatten()
    
    print('Labels Reading Done')
    
    return labels
                         
def read_distribution(path_txt):
    # read txt; channel distribution
    distribution = pandas.read_csv(path_txt, sep='\t')
    
    return distribution

def load_cmdata2d(selected_feature, selected_band, experiment, imshow=False):
    """
    根据选择的特征和频段加载对应的共现矩阵数据。

    Args:
        selected_feature (str): 选择的特征类型 ('PCC' 或 'PLV')。
        selected_band (str): 选择的频段 ('alpha', 'beta', 'gamma', 或 'joint')。

    Returns:
        numpy.ndarray: 根据选择的特征和频段返回的共现矩阵数据。

    Raises:
        ValueError: 当选择的特征或频段无效时抛出。
    """
    if selected_feature == 'PCC':
        cmdata = get_cmdata('PCC', experiment)
        cmdata_alpha = cmdata['alpha']
        cmdata_beta = cmdata['beta']
        cmdata_gamma = cmdata['gamma']
        if selected_band == 'alpha':
            if imshow: draw_projection(cmdata_alpha[0])
            return cmdata_alpha
        elif selected_band == 'beta':
            if imshow: draw_projection(cmdata_beta[0])
            return cmdata_beta
        elif selected_band == 'gamma':
            if imshow: draw_projection(cmdata_gamma[0])
            return cmdata_gamma
        elif selected_band == 'joint':
            cmdata = numpy.stack((cmdata_alpha, cmdata_beta, cmdata_gamma), axis=1)
            if imshow: draw_projection(cmdata[0])
            return cmdata
        else:
            raise ValueError(f"Invalid band selection: {selected_band}")

    elif selected_feature == 'PLV':
        cmdata = get_cmdata('PLV', experiment)
        cmdata_alpha = cmdata['alpha']
        cmdata_beta = cmdata['beta']
        cmdata_gamma = cmdata['gamma']
        if selected_band == 'alpha':
            if imshow: draw_projection(cmdata_alpha[0])
            return cmdata_alpha
        elif selected_band == 'beta':
            if imshow: draw_projection(cmdata_beta[0])
            return cmdata_beta
        elif selected_band == 'gamma':
            if imshow: draw_projection(cmdata_gamma[0])
            return cmdata_gamma
        elif selected_band == 'joint':
            cmdata = numpy.stack((cmdata_alpha, cmdata_beta, cmdata_gamma), axis=1)
            if imshow: draw_projection(cmdata[0])
            return cmdata
        else:
            raise ValueError(f"Invalid band selection: {selected_band}")

    else:
        raise ValueError(f"Invalid feature selection: {selected_feature}")

def load_cmdata1d(selected_feature, selected_band, experiment):
    """
    根据选择的特征和频段加载对应的共现矩阵数据。

    Args:
        selected_feature (str): 选择的特征类型 ('PCC' 或 'PLV')。
        selected_band (str): 选择的频段 ('alpha', 'beta', 'gamma', 或 'joint')。

    Returns:
        numpy.ndarray: 根据选择的特征和频段返回的共现矩阵数据。

    Raises:
        ValueError: 当选择的特征或频段无效时抛出。
    """
    if selected_feature == 'PCC':
        cmdata = get_cmdata('PCC', experiment)
        cmdata_alpha = cmdata['alpha'].reshape(-1, cmdata['alpha'].shape[1]**2)
        cmdata_beta = cmdata['beta'].reshape(-1, cmdata['beta'].shape[1]**2)
        cmdata_gamma = cmdata['gamma'].reshape(-1, cmdata['gamma'].shape[1]**2)
        if selected_band == 'alpha':
            return cmdata_alpha
        elif selected_band == 'beta':
            return cmdata_beta
        elif selected_band == 'gamma':
            return cmdata_gamma
        elif selected_band == 'joint':
            return numpy.hstack((cmdata_alpha, cmdata_beta, cmdata_gamma))
        else:
            raise ValueError(f"Invalid band selection: {selected_band}")

    elif selected_feature == 'PLV':
        cmdata = get_cmdata('PLV', experiment)
        cmdata_alpha = cmdata['alpha'].reshape(-1, cmdata['alpha'].shape[1]**2)
        cmdata_beta = cmdata['beta'].reshape(-1, cmdata['beta'].shape[1]**2)
        cmdata_gamma = cmdata['gamma'].reshape(-1, cmdata['gamma'].shape[1]**2)
        if selected_band == 'alpha':
            return cmdata_alpha
        elif selected_band == 'beta':
            return cmdata_beta
        elif selected_band == 'gamma':
            return cmdata_gamma
        elif selected_band == 'joint':
            return numpy.hstack((cmdata_alpha, cmdata_beta, cmdata_gamma))
        else:
            raise ValueError(f"Invalid band selection: {selected_band}")

    else:
        raise ValueError(f"Invalid feature selection: {selected_feature}")

def get_cmdata(feature, experiment):
    # path
    path_current = os.getcwd()
    path_parent = os.path.dirname(path_current)
    
    # path_data
    path_data = os.path.join(path_parent, 'data', 'SEED', 'functional connectivity', feature, experiment + '.mat')
    
    # cmdata
    cmdata = read_mat(path_data)
    
    return cmdata

def read_mat(path_file):
    # 确保文件存在
    if not os.path.exists(path_file):
        raise FileNotFoundError(f"File not found: {path_file}")

    try:
        # 尝试以 HDF5 格式读取文件
        with h5py.File(path_file, 'r') as f:
            print("HDF5 format detected.")
            # 提取所有键值及其数据
            mat_data = {key: numpy.array(f[key]) for key in f.keys()}

    except OSError:
        # 如果不是 HDF5 格式，尝试使用 scipy.io.loadmat
        print("Not an HDF5 format.")
        mat_data = scipy.io.loadmat(path_file)
        # 排除系统默认的键
        mat_data = {key: mat_data[key] for key in mat_data.keys() if not key.startswith('__')}

    # 数据重塑（如果需要）
    reshaped_data = {key: cmdata_reshaper(data) for key, data in mat_data.items()}

    return reshaped_data

def cmdata_reshaper(mat_data):
    """
    Reshapes mat_data to ensure the last two dimensions are square (n1 == n2).
    Automatically handles transposing and validates the shape.
    """
    MAX_ITER = 10  # 最大迭代次数，防止死循环
    iteration = 0

    while iteration < MAX_ITER:
        if mat_data.ndim == 3:
            samples, n1, n2 = mat_data.shape
            if n1 == n2:
                break  # 如果满足条件，直接退出
            else:
                mat_data = numpy.transpose(mat_data, axes=(2, 0, 1))  # 转置调整维度
        iteration += 1

    else:
        raise ValueError("Failed to reshape mat_data into (samples, n1, n2) with n1 == n2 after multiple attempts.")

    return mat_data

import matplotlib.pyplot as plt

def draw_projection(sample_projection):
    if sample_projection.ndim == 2:
        # Visualize the 2D matrix
        plt.imshow(sample_projection, cmap='viridis')  # Use 'viridis' colormap
        plt.colorbar()  # Add a colorbar
        plt.title("Matrix Visualization using imshow")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.show()
    elif sample_projection.ndim == 3 and sample_projection.shape[0] == 3:
        # Visualize each channel of the 3D projection
        for i, channel_projection in enumerate(sample_projection):
            plt.imshow(channel_projection, cmap='viridis')  # Use 'viridis' colormap
            plt.colorbar()  # Add a colorbar
            plt.title(f"Matrix Visualization of Channel {i + 1}")
            plt.xlabel("X-axis")
            plt.ylabel("Y-axis")
            plt.show()
    else:
        raise ValueError("Input projection sample should be a 2D array or a 3D array with 3 channels.")

# %% Example Usage
if __name__ == '__main__':
    cmdata1d_joint = load_cmdata1d('PLV', 'joint', 'sub1ex1')
    cmdata1d_gamma = load_cmdata1d('PLV', 'gamma', 'sub1ex1')
    cmdata2d_joint = load_cmdata2d('PLV', 'joint', 'sub1ex1')
    cmdata2d_gamma = load_cmdata2d('PLV', 'gamma', 'sub1ex1')
    
    