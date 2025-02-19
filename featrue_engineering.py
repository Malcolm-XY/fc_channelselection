# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 23:15:11 2025

@author: 18307
"""

import os
import time
import pickle
import numpy as np
import pandas as pd

import mne
import mne_connectivity
from scipy.signal import hilbert
from scipy.stats import gaussian_kde

import joblib
import utils

# %% Filter EEG
def filter_eeg(eeg, freq=128, verbose=False):
    info = mne.create_info(ch_names=['Ch' + str(i) for i in range(eeg.shape[0])], sfreq=freq, ch_types='eeg')
    mneeeg = mne.io.RawArray(eeg, info)
    
    freq_bands = {
        "Delta": (0.5, 4),
        "Theta": (4, 8),
        "Alpha": (8, 13),
        "Beta": (13, 30),
        "Gamma": (30, 63),
    }
    
    band_filtered_eeg = {}

    for band, (low_freq, high_freq) in freq_bands.items():
        filtered_eeg = mneeeg.copy().filter(l_freq=low_freq, h_freq=high_freq, method="fir", phase="zero-double")
        band_filtered_eeg[band] = filtered_eeg
        if verbose:
            print(f"{band} band filtered: {low_freq}–{high_freq} Hz")

    return band_filtered_eeg

def filter_eeg_and_save_seed(subject, experiment, verbose=True):
    eeg = utils.load_seed(subject, experiment)
    
    path_current = os.getcwd()
    path_parent = os.path.dirname(path_current)
    path_folder = os.path.join(path_parent, 'data', 'SEED', 'original eeg', 'Filtered_EEG')
    os.makedirs(path_folder, exist_ok=True)
    
    # 调用 filter_eeg 函数
    filtered_eeg_dict = filter_eeg(eeg, verbose=verbose)
    
    # 保存每个频段的数据
    for band, filtered_eeg in filtered_eeg_dict.items():
        path_file = os.path.join(path_folder, f"sub{subject+1}_{band}_eeg.fif")
        filtered_eeg.save(path_file, overwrite=True)
        if verbose:
            print(f"Saved {band} band filtered EEG to {path_file}")
    
    return filtered_eeg_dict

def filter_eeg_and_save_dreamer(subject, verbose=True):
    _, eeg, _ = utils.load_dreamer()
    eeg = eeg[subject].transpose()
    
    path_current = os.getcwd()
    path_parent = os.path.dirname(path_current)
    path_folder = os.path.join(path_parent, 'data', 'DREAMER', 'Filtered_EEG')
    os.makedirs(path_folder, exist_ok=True)
    
    # 调用 filter_eeg 函数
    filtered_eeg_dict = filter_eeg(eeg, verbose=verbose)
    
    # 保存每个频段的数据
    for band, filtered_eeg in filtered_eeg_dict.items():
        path_file = os.path.join(path_folder, f"sub{subject+1}_{band}_eeg.fif")
        filtered_eeg.save(path_file, overwrite=True)
        if verbose:
            print(f"Saved {band} band filtered EEG to {path_file}")
    
    return filtered_eeg_dict
                
def read_filtered_eegdata_dreamer(identifier, freq_band="Joint"):
    """
    Read filtered EEG data for the specified experiment and frequency band.

    Parameters:
    experiment (str): Name of the experiment (e.g., subject or session).
    freq_band (str): Frequency band to load ("alpha", "beta", "gamma", "delta", "theta", or "joint").
                     Default is "Joint".

    Returns:
    mne.io.Raw | dict: Returns the MNE Raw object for a single band or a dictionary of Raw objects for "joint".

    Raises:
    ValueError: If the specified frequency band is not valid.
    FileNotFoundError: If the expected file does not exist.
    """
    path_current = os.getcwd()
    path_parent = os.path.dirname(path_current)
    path_folder = os.path.join(path_parent, 'data', 'DREAMER', 'Filtered_EEG')

    try:
        if freq_band in ["alpha", "beta", "gamma", "delta", "theta"]:
            path_file = os.path.join(path_folder, f"{identifier}_{freq_band.capitalize()}_eeg.fif")
            filtered_eeg = mne.io.read_raw_fif(path_file, preload=True)
            return filtered_eeg

        elif freq_band.lower() == "joint":
            filtered_eeg = {}
            for band in ["Alpha", "Beta", "Gamma", "Delta", "Theta"]:
                path_file = os.path.join(path_folder, f"{identifier}_{band}_eeg.fif")
                filtered_eeg[band.lower()] = mne.io.read_raw_fif(path_file, preload=True)
            return filtered_eeg

        else:
            raise ValueError(f"Invalid frequency band: {freq_band}. Choose from 'alpha', 'beta', 'gamma', 'delta', 'theta', or 'joint'.")

    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found for '{identifier}' and frequency band '{freq_band}'. Check the path and file existence.")

# %% Feature Engineering
def compute_distance_matrix():
    distribution = utils.get_distribution()
    electrod_names, x, y, z = distribution['channel'], distribution['x'], distribution['y'], distribution['z']
    
    # 将 x, y, z 坐标堆叠成 (N, 3) 形状的矩阵
    coords = np.vstack((x, y, z)).T  # 形状 (N, 3)
    
    # 计算欧几里得距离矩阵
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    distance_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))
    
    return distance_matrix

def fc_matrices_circle(dataset, feature='pcc', subject_range=range(1, 2), experiment_range=range(1, 2), freq_band='joint', save=False, verbose=True):
    """
    计算 SEED 数据集的相关矩阵，并可选保存为 pickle 文件。
    
    **新增功能**:
    - 记录总时间
    - 记录每个 experiment 的平均时间

    参数：
    dataset (str): 数据集名称（目前仅支持 'SEED'）。
    subject_range (range): 被试 ID 范围，默认 1~2。
    experiment_range (range): 实验 ID 范围，默认 1~2。
    freq_band (str): 频带类型，可选 'alpha', 'beta', 'gamma' 或 'joint'（默认）。
    save (bool): 是否保存结果，默认 False。
    verbose (bool): 是否打印计时信息，默认 True。

    返回：
    dict: 计算得到的相关矩阵字典。
    """
    if dataset.lower() != 'seed':
        raise ValueError("当前函数仅支持 SEED 数据集")

    fc_matrices_dict = {}

    # **开始计时**
    start_time = time.time()
    experiment_count = 0  # 计数 experiment 计算次数
    total_experiment_time = 0  # 累计 experiment 计算时间

    for subject in subject_range:
        for experiment in experiment_range:
            experiment_start_time = time.time()  # 记录单次 experiment 开始时间
            experiment_count += 1

            identifier = f'sub{subject}ex{experiment}'
            eeg_data = utils.load_seed_filtered(subject, experiment)

            if freq_band.lower() in ['alpha', 'beta', 'gamma']:
                data = np.array(eeg_data[freq_band.lower()])
                if feature.lower() == 'pcc': 
                    fc_matrices_dict[identifier] = compute_corr_matrices(data, samplingrate=200)
                elif feature.lower() == 'plv': 
                    fc_matrices_dict[identifier] = compute_plv_matrices(data, samplingrate=200)
                elif feature.lower() == 'mi': 
                    fc_matrices_dict[identifier] = compute_mi_matrices(data, samplingrate=200)

            elif freq_band.lower() == 'joint':
                fc_matrices_dict[identifier] = {}  # 确保是字典
                for band in ['alpha', 'beta', 'gamma']:
                    data = np.array(eeg_data[band])
                    if feature.lower() == 'pcc': 
                        fc_matrices_dict[identifier][band] = compute_corr_matrices(data, samplingrate=200)
                    elif feature.lower() == 'plv': 
                        fc_matrices_dict[identifier][band] = compute_plv_matrices(data, samplingrate=200)
                    elif feature.lower() == 'mi': 
                        fc_matrices_dict[identifier][band] = compute_mi_matrices(data, samplingrate=200)

            # **记录单个 experiment 计算时间**
            experiment_time = time.time() - experiment_start_time
            total_experiment_time += experiment_time
            if verbose:
                print(f"Experiment {identifier} completed in {experiment_time:.2f} seconds")

            # **保存计算结果**
            if save:
                path_current = os.getcwd()
                path_parent = os.path.dirname(path_current)
                path_parent_parent = os.path.dirname(path_parent)
                
                path_folder = os.path.join(path_parent_parent, 'Research_Data', 'SEED', 'functional connectivity', f'{feature}_pkl')

                # 确保目标文件夹存在
                os.makedirs(path_folder, exist_ok=True)

                if freq_band.lower() == 'joint':
                    for band in ['alpha', 'beta', 'gamma']:
                        file_path_band_pkl = os.path.join(path_folder, f"{identifier}_{band}.pkl")
                        with open(file_path_band_pkl, 'wb') as f:
                            pickle.dump(fc_matrices_dict[identifier][band], f)
                else:
                    file_path_pkl = os.path.join(path_folder, f"{identifier}_{freq_band.lower()}.pkl")
                    with open(file_path_pkl, 'wb') as f:
                        pickle.dump(fc_matrices_dict[identifier], f)

    # **计算总时间 & 平均 experiment 时间**
    total_time = time.time() - start_time
    avg_experiment_time = total_experiment_time / experiment_count if experiment_count > 0 else 0

    if verbose:
        print(f"\nTotal time taken: {total_time:.2f} seconds")
        print(f"Average time per experiment: {avg_experiment_time:.2f} seconds")

    return fc_matrices_dict

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
        utils.draw_projection(avg_corr_matrix)

    return corr_matrices

def compute_plv_matrices(eeg_data, samplingrate, window=1, overlap=0, verbose=True, visualization=True):
    """
    Compute Phase Locking Value (PLV) matrices for EEG data using a sliding window approach.
    
    Parameters:
        eeg_data (numpy.ndarray): EEG data with shape (channels, time_samples).
        samplingrate (int): Sampling rate of the EEG data in Hz.
        window (float): Window size in seconds for segmenting EEG data.
        overlap (float): Overlap fraction between consecutive windows (0 to 1).
        verbose (bool): If True, prints progress.
        visualization (bool): If True, displays PLV matrices.
    
    Returns:
        list of numpy.ndarray: List of PLV matrices for each window.
    """
    step = int(samplingrate * window * (1 - overlap))  # Step size for moving window
    segment_length = int(samplingrate * window)

    # Split EEG data into overlapping windows
    split_segments = [
        eeg_data[:, i:i + segment_length] 
        for i in range(0, eeg_data.shape[1] - segment_length + 1, step)
    ]

    plv_matrices = []
    for idx, segment in enumerate(split_segments):
        if segment.shape[1] < segment_length:
            continue  # Skip incomplete segments
        
        # Compute Hilbert transform to obtain instantaneous phase
        analytic_signal = hilbert(segment, axis=1)
        phase_data = np.angle(analytic_signal)  # Extract phase information
        
        # Compute PLV matrix
        num_channels = phase_data.shape[0]
        plv_matrix = np.zeros((num_channels, num_channels))
        
        for ch1 in range(num_channels):
            for ch2 in range(num_channels):
                phase_diff = phase_data[ch1, :] - phase_data[ch2, :]
                plv_matrix[ch1, ch2] = np.abs(np.mean(np.exp(1j * phase_diff)))
        
        plv_matrices.append(plv_matrix)

        if verbose:
            print(f"Computed PLV matrix {idx + 1}/{len(split_segments)}")
    
    # Optional visualization
    if visualization and plv_matrices:
        avg_plv_matrix = np.mean(plv_matrices, axis=0)
        utils.draw_projection(avg_plv_matrix)
    
    return plv_matrices

# %% MI computing
from tqdm import tqdm  # 用于进度条显示
def compute_mi_matrices(eeg_data, samplingrate, window=1, overlap=0, verbose=True, visualization=True):
    """
    Compute Mutual Information (MI) matrices for EEG data using a sliding window approach (optimized with parallelism).

    Parameters:
        eeg_data (numpy.ndarray): EEG data with shape (channels, time_samples).
        samplingrate (int): Sampling rate of the EEG data in Hz.
        window (float): Window size in seconds for segmenting EEG data.
        overlap (float): Overlap fraction between consecutive windows (0 to 1).
        verbose (bool): If True, prints progress.
        visualization (bool): If True, displays MI matrices.

    Returns:
        list of numpy.ndarray: List of MI matrices for each window.
    """
    if verbose:
        print("Starting Mutual Information computation...")
    
    step = int(samplingrate * window * (1 - overlap))  # Step size for moving window
    segment_length = int(samplingrate * window)

    if verbose:
        print("Segmenting EEG data...")
    # Split EEG data into overlapping windows
    split_segments = [
        eeg_data[:, i:i + segment_length]
        for i in range(0, eeg_data.shape[1] - segment_length + 1, step)
    ]

    num_windows = len(split_segments)
    if verbose:
        print(f"Total segments: {num_windows}")

    def compute_mi_matrix(segment):
        """ Compute MI matrix for a single segment (Parallelizable). """
        num_channels = segment.shape[0]
        mi_matrix = np.zeros((num_channels, num_channels))

        def compute_mi(x, y):
            """ Fast mutual information computation using histogram method. """
            hist_2d, _, _ = np.histogram2d(x, y, bins=5)
            pxy = hist_2d / np.sum(hist_2d)
            px = np.sum(pxy, axis=1)
            py = np.sum(pxy, axis=0)
            px_py = np.outer(px, py)
            nonzero = pxy > 0  # Avoid log(0)
            return np.sum(pxy[nonzero] * np.log(pxy[nonzero] / px_py[nonzero]))
        
        # Parallel computation of MI matrix (only upper triangle)
        mi_values = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(compute_mi)(segment[ch1], segment[ch2])
            for ch1 in range(num_channels) for ch2 in range(ch1 + 1, num_channels)
        )

        # Fill the matrix symmetrically
        idx = 0
        for ch1 in range(num_channels):
            for ch2 in range(ch1 + 1, num_channels):
                mi_matrix[ch1, ch2] = mi_matrix[ch2, ch1] = mi_values[idx]
                idx += 1

        np.fill_diagonal(mi_matrix, 1)  # Self-MI is 1
        return mi_matrix

    if verbose:
        print("Computing MI matrices...")

    # Compute MI matrices in parallel with progress tracking
    # mi_matrices = joblib.Parallel(n_jobs=8, verbose=10)(
    #     joblib.delayed(compute_mi_matrix)(segment) for segment in split_segments
    # )
    
    mi_matrices = []
    for segment in tqdm(split_segments, desc="Processing segments", disable=not verbose):
        mi_matrices.append(compute_mi_matrix(segment))
    
    if verbose:
        print(f"Computed {len(mi_matrices)} MI matrices.")

    # Optional visualization
    if visualization and mi_matrices:
        avg_mi_matrix = np.mean(mi_matrices, axis=0)
        utils.draw_projection(avg_mi_matrix)

    return mi_matrices

# %% Label Engineering
def read_labels_dreamer():
    path_current = os.getcwd()
    path_parent = os.path.dirname(path_current)
    path_labels = os.path.join(path_parent, 'data', 'DREAMER', 'labels', 'labels.txt')
    df = pd.read_csv(path_labels, sep=r'\s+', engine='python')
    return {col: df[col].to_numpy() for col in df.columns}

def generate_labels(samplingrate=128):
    path_current = os.getcwd()
    path_parent = os.path.dirname(path_current)
    path_data = os.path.join(path_parent, 'data', 'DREAMER', 'DREAMER.mat')

    mat_data = utils.read_mat(path_data)
    
    # %% labels
    score_arousal = 0
    score_dominance = 0
    score_valence = 0
    index = 0
    eeg_all = []
    for data in mat_data['DREAMER']['Data']:
        index += 1
        score_arousal += data['ScoreArousal']
        score_dominance += data['ScoreDominance']
        score_valence += data['ScoreValence']
        eeg_all.append(data['EEG']['stimuli'])
        
    labels = [1, 3, 5]
    score_arousal_labels = normalize_to_labels(score_arousal, labels)
    score_dominance_labels = normalize_to_labels(score_dominance, labels)
    score_valence_labels = normalize_to_labels(score_valence, labels)
    
    # %% data
    eeg_sample = eeg_all[0]
    labels_arousal = []
    labels_dominance = []
    labels_valence = []
    for eeg_trial in range(0,len(eeg_sample)):     
        label_container = np.ones(len(eeg_sample[eeg_trial]))
        
        label_arousal = label_container * score_arousal_labels[eeg_trial]
        label_dominance = label_container * score_dominance_labels[eeg_trial]
        label_valence = label_container * score_valence_labels[eeg_trial]
        
        labels_arousal = np.concatenate((labels_arousal, label_arousal))
        labels_dominance = np.concatenate((labels_dominance, label_dominance))
        labels_valence = np.concatenate((labels_valence, label_valence))
        
    labels_arousal = labels_arousal[::samplingrate]
    labels_dominance = labels_dominance[::samplingrate]
    labels_valence = labels_valence[::samplingrate]
    
    return labels_arousal, labels_dominance, labels_valence

def normalize_to_labels(array, labels):
    """
    Normalize an array to discrete labels.
    
    Parameters:
        array (np.ndarray): The input array.
        labels (list): The target labels to map to (e.g., [1, 3, 5]).
    
    Returns:
        np.ndarray: The normalized array mapped to discrete labels.
    """
    # Step 1: Normalize array to [0, 1]
    array_min = np.min(array)
    array_max = np.max(array)
    normalized = (array - array_min) / (array_max - array_min)
    
    # Step 2: Map to discrete labels
    bins = np.linspace(0, 1, len(labels))
    discrete_labels = np.digitize(normalized, bins, right=True)
    
    # Map indices to corresponding labels
    return np.array([labels[i - 1] for i in discrete_labels])

# %% interpolation
import scipy.interpolate
def interpolate_matrices(data, scale_factor=(1.0, 1.0), method='nearest'):
    """
    对形如 samples x channels x w x h 的数据进行插值，使每个 w x h 矩阵放缩

    参数:
    - data: numpy.ndarray, 形状为 (samples, channels, w, h)
    - scale_factor: float 或 (float, float)，插值的缩放因子
    - method: str，插值方法，可选：
        - 'nearest' (最近邻)
        - 'linear' (双线性)
        - 'cubic' (三次插值)

    返回:
    - new_data: numpy.ndarray, 插值后的数据，形状 (samples, channels, new_w, new_h)
    """
    samples, channels, w, h = data.shape
    new_w, new_h = int(w * scale_factor[0]), int(h * scale_factor[1])

    # 目标尺寸
    output_shape = (samples, channels, new_w, new_h)
    new_data = np.zeros(output_shape, dtype=data.dtype)

    # 原始网格点 (w, h)
    x_old = np.linspace(0, 1, w)
    y_old = np.linspace(0, 1, h)
    xx_old, yy_old = np.meshgrid(x_old, y_old, indexing='ij')

    # 目标网格点 (new_w, new_h)
    x_new = np.linspace(0, 1, new_w)
    y_new = np.linspace(0, 1, new_h)
    xx_new, yy_new = np.meshgrid(x_new, y_new, indexing='ij')

    # 插值
    for i in range(samples):
        for j in range(channels):
            old_points = np.column_stack([xx_old.ravel(), yy_old.ravel()])  # 原始点坐标
            new_points = np.column_stack([xx_new.ravel(), yy_new.ravel()])  # 目标点坐标
            values = data[i, j].ravel()  # 原始像素值

            # griddata 进行插值
            interpolated = scipy.interpolate.griddata(old_points, values, new_points, method=method)
            new_data[i, j] = interpolated.reshape(new_w, new_h)

    return new_data

# %% usage
if __name__ == "__main__":
    # distance_matrix = compute_distance_matrix()
    # utils.draw_projection(distance_matrix)
    
    # fc_pcc_matrices = fc_matrices_circle('SEED', feature='pcc', save=True, subject_range=range(1, 16), experiment_range=range(1, 4))
    # fc_plv_matrices = fc_matrices_circle('SEED', feature='plv', save=True, subject_range=range(1, 16), experiment_range=range(1, 4))
    fc_mi_matrices = fc_matrices_circle('SEED', feature='mi', save=True, subject_range=range(15, 16), experiment_range=range(1, 4))
    
    # %% End program actions
    utils.end_program_actions(play_sound=True, shutdown=False, countdown_seconds=120)