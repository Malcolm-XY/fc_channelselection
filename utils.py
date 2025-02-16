# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:43:46 2025

@author: usouu
"""

import os
import h5py

import numpy as np
import pandas as pd

import scipy.io
import scipy.ndimage
import matplotlib.pyplot as plt

import mne

# %% Common Functions
def read_mat(path_file, simplify=True):
    """
    读取 MATLAB 的 .mat 文件。
    - 自动支持 HDF5 格式和非 HDF5 格式。
    - 可选简化数据结构。
    
    参数：
    - path_file (str): .mat 文件路径。
    - simplify (bool): 是否简化数据结构（默认 True）。
    
    返回：
    - dict: 包含 .mat 文件数据的字典。
    """
    # 确保文件存在
    if not os.path.exists(path_file):
        raise FileNotFoundError(f"File not found: {path_file}")

    try:
        # 尝试以 HDF5 格式读取文件
        with h5py.File(path_file, 'r') as f:
            print("HDF5 format detected.")
            data = {key: simplify_mat_structure(f[key]) for key in f.keys()} if simplify else f
            return data

    except OSError:
        # 如果不是 HDF5 格式，尝试使用 scipy.io.loadmat
        print("Not an HDF5 format.")
        mat_data = scipy.io.loadmat(path_file, squeeze_me=simplify, struct_as_record=not simplify)
        if simplify:
            mat_data = {key: simplify_mat_structure(value) for key, value in mat_data.items() if key[0] != '_'}
        return mat_data

def simplify_mat_structure(data):
    """
    递归解析和简化 MATLAB 数据结构。
    - 将结构体转换为字典。
    - 将 Cell 数组转换为列表。
    - 移除 NumPy 数组中的多余维度。
    """
    if isinstance(data, h5py.Dataset):  # 处理 HDF5 数据集
        return data[()]  # 转换为 NumPy 数组或标量

    elif isinstance(data, h5py.Group):  # 处理 HDF5 文件组
        return {key: simplify_mat_structure(data[key]) for key in data.keys()}

    elif isinstance(data, scipy.io.matlab.mat_struct):  # 处理 MATLAB 结构体
        return {field: simplify_mat_structure(getattr(data, field)) for field in data._fieldnames}

    elif isinstance(data, np.ndarray):  # 处理 NumPy 数组
        if data.dtype == 'object':  # 递归解析对象数组
            return [simplify_mat_structure(item) for item in data]
        return data.squeeze()  # 移除多余维度

    else:  # 其他类型直接返回
        return data

def read_filtered_eegdata(folder, identifier, freq_band='joint', object_type='pandas_dataframe'):
    """
    Read filtered EEG data for the specified experiment and frequency band.

    Parameters:
    folder (str): Directory containing the EEG data files.
    identifier (str): Identifier for the subject/session.
    freq_band (str): Frequency band to load ("alpha", "beta", "gamma", "delta", "theta", or "joint").
                     Default is "joint", which loads all bands.
    object_type (str): Desired output format: 'pandas_dataframe', 'numpy_array', or 'mne'.

    Returns:
    mne.io.Raw | dict | pandas.DataFrame | numpy.ndarray:
        - If 'mne', returns the MNE Raw object (or a dictionary of them for 'joint').
        - If 'pandas_dataframe', returns a DataFrame with EEG data.
        - If 'numpy_array', returns a NumPy array with EEG data.

    Raises:
    ValueError: If the specified frequency band is not valid.
    FileNotFoundError: If the expected file does not exist.
    """

    try:
        if freq_band.lower() in ['alpha', 'beta', 'gamma', 'delta', 'theta']:
            path_file = os.path.join(folder, f'{identifier}_{freq_band.capitalize()}_eeg.fif')
            filtered_eeg = mne.io.read_raw_fif(path_file, preload=True)

            if object_type == 'pandas_dataframe':
                return pd.DataFrame(filtered_eeg.get_data(), index=filtered_eeg.ch_names)
            elif object_type == 'numpy_array':
                return filtered_eeg.get_data()
            else:
                return filtered_eeg

        elif freq_band.lower() == 'joint':
            filtered_eeg = {}
            for band in ['alpha', 'beta', 'gamma', 'delta', 'theta']:
                path_file = os.path.join(folder, f'{identifier}_{band}_eeg.fif')
                raw_data = mne.io.read_raw_fif(path_file, preload=True)

                if object_type == 'pandas_dataframe':
                    filtered_eeg[band.lower()] = pd.DataFrame(raw_data.get_data(), index=raw_data.ch_names)
                elif object_type == 'numpy_array':
                    filtered_eeg[band.lower()] = raw_data.get_data()
                else:
                    filtered_eeg[band.lower()] = raw_data

            return filtered_eeg

        else:
            raise ValueError(f"Invalid frequency band: {freq_band}. Choose from 'alpha', 'beta', 'gamma', 'delta', 'theta', or 'joint'.")

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found for '{identifier}' and frequency band '{freq_band}'. Check the path and file existence.")

def draw_projection(sample_projection):
    """
    Visualizes data projections (common for both datasets).
    """
    if sample_projection.ndim == 2:
        plt.imshow(sample_projection, cmap='viridis')
        plt.colorbar()
        plt.title("2D Matrix Visualization")
        plt.show()
    elif sample_projection.ndim == 3 and sample_projection.shape[0] == 3:
        for i in range(3):
            plt.imshow(sample_projection[i], cmap='viridis')
            plt.colorbar()
            plt.title(f"Channel {i + 1} Visualization")
            plt.show()

def read_labels(dataset='SEED'):
    if dataset == 'SEED':
        labels = read_labels_seed()
    elif dataset == 'DREAMER':
        labels = read_labels_dreamer()
    return labels

def get_distribution(mapping_method='auto'):
    # define path
    path_current = os.getcwd()
    path_parent = os.path.dirname(path_current)
    path_distribution = os.path.join(path_parent, 'data', 'SEED', 'electrode distribution')
        
    # read distribution txt
    if mapping_method == 'auto':
        path_ch_auto_distr = os.path.join(path_distribution, 
                                          'biosemi62_64_channels_original_distribution.txt')
        # read txt; channel distribution
        distribution = pd.read_csv(path_ch_auto_distr, sep='\t')
        
    elif mapping_method == 'manual':
        path_ch_manual_distr = os.path.join(path_distribution, 
                                            'biosemi62_64_channels_manual_distribution.txt')    
        # read txt; channel distribution
        distribution = pd.read_csv(path_ch_manual_distr, sep='\t')
    
    return distribution

# %% SEED Specific Functions
# original eegl; seed
def load_seed(subject, experiment, band='full'):
    path_current = os.getcwd()
    path_parent = os.path.dirname(path_current)
    path_1 = os.path.join(path_parent, 'data', 'SEED', 'original eeg')
    path_2 = os.path.join(path_1, 'Preprocessed_EEG')
    path_3 = os.path.join(path_1, 'Filtered_EEG')
    
    identifier = f'sub{subject}ex{experiment}'
    
    if band == 'full':
        path_data = os.path.join(path_2, identifier + '.mat')
        data_temp = read_mat(path_data)
        
        data_list = [data_temp[dat] for dat in data_temp]
        data = np.hstack(data_list)
        
    else:
        data = read_filtered_eegdata(path_3, identifier, band)

    return data

def load_seed_filtered(subject, experiment, band='joint'):
    path_current = os.getcwd()
    path_parent = os.path.dirname(path_current)
    folder = os.path.join(path_parent, 'data', 'SEED', 'original eeg', 'Filtered_EEG')
    
    identifier = f'sub{experiment}ex{experiment}'
    
    data = read_filtered_eegdata(folder, identifier, freq_band=band)
    return data

def load_cms_seed(experiment, feature='PCC', band='joint', imshow=True):
    path_current = os.getcwd()
    path_parent = os.path.dirname(path_current)
    path_data = os.path.join(path_parent, 'data', 'SEED', 'functional connectivity', feature, f"{experiment}.mat")
    cms = read_mat(path_data)
    
    cms_alpha = cms['alpha']
    cms_beta = cms['beta']
    cms_gamma = cms['gamma']
    
    if band == 'joint':
        data = np.stack((cms_alpha, cms_beta, cms_gamma), axis=1)
    else:
        data = cms[band]

    if imshow:
       draw_projection(np.mean(data, axis=0))
    
    return data

def read_labels_seed():
    path_current = os.getcwd()
    path_parent = os.path.dirname(path_current)
    path_labels = os.path.join(path_parent, 'data', 'SEED', 'labels', 'labels.txt')
    return pd.read_csv(path_labels, sep='\t', header=None).to_numpy().flatten()

# %% DREAMER Specific Functions
# original eeg; dreamer
def load_dreamer():
    path_current = os.getcwd()
    path_parent = os.path.dirname(path_current)
    path_data = os.path.join(path_parent, 'data', 'DREAMER', 'DREAMER.mat')
    dreamer = read_mat(path_data)
    eeg_dic = [np.vstack(trial["EEG"]["stimuli"]) for trial in dreamer["DREAMER"]["Data"]]
    return dreamer, eeg_dic, dreamer["DREAMER"]["EEG_Electrodes"]

def load_dreamer_filtered(experiment, band='joint'):
    path_current = os.getcwd()
    path_parent = os.path.dirname(path_current)
    folder = os.path.join(path_parent, 'data', 'DREAMER', 'Filtered_EEG')
    
    identifier = f'sub{experiment}'
    
    data = read_filtered_eegdata(folder, identifier, freq_band=band)
    return data

def load_cms_dreamer(experiment, feature='PCC', band='joint', imshow=True):
    # 获取当前路径及父路径
    path_current = os.getcwd()
    path_parent = os.path.dirname(path_current)
    
    # 根据方法选择对应文件夹
    if feature == 'PCC':
        path_folder = os.path.join(path_parent, 'data', 'DREAMER', 'functional connectivity', 'PCC')
    elif feature == 'PLV':
        path_folder = os.path.join(path_parent, 'data', 'DREAMER', 'functional connectivity', 'PLV')
    else:
        raise ValueError(f"Unsupported feature: {feature}")
    
    # 拼接数据文件路径
    path_file = os.path.join(path_folder, f'{experiment}.npy')
    
    # 加载数据
    cms_load = np.load(path_file, allow_pickle=True).item()
    
    # 从加载的数据中获取各频段的列表（列表中每个元素为形状 wxw 的数组）
    cms_alpha = cms_load["alpha"]
    cms_beta = cms_load["beta"]
    cms_gamma = cms_load["gamma"]
    
    # 将列表转换为 NumPy 数组，形状为 (n_samples, w, w)
    cms_alpha = np.array(cms_alpha)
    cms_beta = np.array(cms_beta)
    cms_gamma = np.array(cms_gamma)     
        
    # 根据 freq_band 参数返回相应的数据
    if band == "alpha":
        if imshow: draw_projection(np.mean(cms_alpha, axis=0))
        return cms_alpha
    elif band == "beta":
        if imshow: draw_projection(np.mean(cms_beta, axis=0))
        return cms_beta
    elif band == "gamma":
        if imshow: draw_projection(np.mean(cms_gamma, axis=0))
        return cms_gamma
    elif band == "joint":
        joint = np.stack([cms_alpha, cms_beta, cms_gamma], axis=1)
        if imshow: draw_projection(np.mean(joint, axis=0))
        return joint
    else:
        raise ValueError(f"Unknown freq_band parameter: {band}")

def read_labels_dreamer():
    path_current = os.getcwd()
    path_parent = os.path.dirname(path_current)
    path_labels = os.path.join(path_parent, 'data', 'DREAMER', 'labels', 'labels.txt')
    df = pd.read_csv(path_labels, sep=r'\s+', engine='python')
    return {col: df[col].to_numpy() for col in df.columns}

def normalize_to_labels(array, labels):
    normalized = (array - np.min(array)) / (np.max(array) - np.min(array))
    bins = np.linspace(0, 1, len(labels))
    return np.array([labels[np.digitize(val, bins) - 1] for val in normalized])

# %% Dataset Selector
def load_dataset(dataset='SEED', **kwargs):
    if dataset == 'SEED':
        return load_seed(**kwargs)
        # return None
    elif dataset == 'DREAMER':
        return load_dreamer(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
        
def load_cms(dataset='SEED', **kwargs):
    if dataset == 'SEED':
        return load_cms_seed(**kwargs)
    elif dataset == 'DREAMER':
        return load_cms_dreamer(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

# %% Example Usage
if __name__ == '__main__':
    # %% SEED
    # dataset
    seed_sub_sample, seed_ex_sample, seed_fre_sample1, seed_fre_sample2  = 1, 1, 'split', 'full'
    seed_sample_1 = load_seed(seed_sub_sample, seed_ex_sample, band=seed_fre_sample1)
    seed_sample_2 = load_seed(seed_sub_sample, seed_ex_sample, band=seed_fre_sample2)
    
    # cms
    experiment_sample, feature_sample, freq_sample_1, freq_sample_2 = 'sub1ex1', 'PCC', 'alpha', 'joint'
    seed_cms_sample_1 = load_cms(dataset='SEED', experiment=experiment_sample, feature=feature_sample, band=freq_sample_1)
    seed_cms_sample_2 = load_cms(dataset='SEED', experiment=experiment_sample, feature=feature_sample, band=freq_sample_2)
    
    labels_SEED = read_labels_seed()
    
    # %% DREAMER
    # dataset
    dreamer, dreamer_eeg_, dreamer_electrodes = load_dataset(dataset='DREAMER')
    dreamer_sub_sample, dreamer_fre_sample_1, dreamer_fre_sample_2 = 1, 'alpha', 'joint'
    draemer_sample_1 = load_dreamer_filtered(dreamer_sub_sample, band=dreamer_fre_sample_1)
    draemer_sample_2 = load_dreamer_filtered(dreamer_sub_sample, band=dreamer_fre_sample_2)
    
    # cms
    experiment_sample_d, feature_sample_d, freq_sample_d1, freq_sample_d2 = 'sub1', 'PCC', 'alpha', 'joint'
    dreamer_cms_sample_1 = load_cms(dataset='DREAMER', experiment=experiment_sample_d, feature=feature_sample_d, band=freq_sample_d1)
    dreamer_cms_sample_1 = load_cms(dataset='DREAMER', experiment=experiment_sample_d, feature=feature_sample_d, band=freq_sample_d2)
    
    labels_DREAMER = read_labels_dreamer()
