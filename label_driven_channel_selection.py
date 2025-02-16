# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 23:43:14 2025

@author: 18307
"""

import utils
import numpy as np

def compute_mi(x, y):
    """ Fast mutual information computation using histogram method. """
    hist_2d, _, _ = np.histogram2d(x, y, bins=5)
    pxy = hist_2d / np.sum(hist_2d)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = np.outer(px, py)
    nonzero = pxy > 0  # Avoid log(0)
    return np.sum(pxy[nonzero] * np.log(pxy[nonzero] / px_py[nonzero]))

def min_max_normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

import seaborn as sns
import matplotlib.pyplot as plt
def plot_heatmap(data, xticklabels=None):
    """
    Plots a heatmap for a 1xN array.

    Parameters:
        data (numpy.ndarray): 1xN array for visualization.
        xticklabels (list, optional): Labels for the x-axis.
    """
    plt.figure(figsize=(10, 2))
    sns.heatmap(
        data, 
        cmap='Blues',  # 单色热图
        annot=False,  # 不显示具体数值
        linewidths=0.5, 
        xticklabels=xticklabels, 
        yticklabels=False
    )
    plt.title("Heatmap of 1xN Array")
    plt.show()

def downsample_mean(eeg_sample, factor=200):
    channels, points = eeg_sample.shape
    truncated_length = points - (points % factor)  # 确保整除
    eeg_trimmed = eeg_sample[:, :truncated_length]  # 截断到可整除的长度
    eeg_downsampled = eeg_trimmed.reshape(channels, -1, factor).mean(axis=2)  # 每 factor 组取平均值
    return eeg_downsampled

from scipy.signal import decimate
def downsample_decimate(eeg_sample, factor=200):
    return decimate(eeg_sample, factor, axis=1, ftype='fir', zero_phase=True)

labels = utils.read_labels()
eeg_samples = utils.load_dataset('SEED', subject=1, experiment=1)

# 使用函数
eeg_samples_downsampled = downsample_mean(eeg_samples, factor=200)

mis = []
for eeg in eeg_samples_downsampled:
    mi = compute_mi(eeg, labels)
    mis.append(mi)

mis = np.array(mis)
mis = mis.reshape(1, -1)

normalized_mi = min_max_normalize(mis)

distribution = utils.get_distribution()
electrodes = distribution['channel']

plot_heatmap(normalized_mi, electrodes)