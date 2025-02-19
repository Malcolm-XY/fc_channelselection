# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 23:43:14 2025

@author: 18307
"""

import utils
import numpy as np
import pandas as pd
import scipy.signal

def compute_mi(x, y):
    """ Fast mutual information computation using histogram method. """
    hist_2d, _, _ = np.histogram2d(x, y, bins=5)
    pxy = hist_2d / np.sum(hist_2d)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = np.outer(px, py)
    nonzero = pxy > 0  # Avoid log(0)
    return np.sum(pxy[nonzero] * np.log(pxy[nonzero] / px_py[nonzero]))

def compute_mis(xs, y, verbose=True):
    mis = []
    for x in xs:
        mi = compute_mi(x, y)
        mis.append(mi)
    
    mis = np.array(mis,  dtype=float)
    normalized_mis = min_max_normalize(mis)

    if verbose:
        distribution = utils.get_distribution()
        electrodes = distribution['channel']
        
        plot_heatmap(mis, electrodes)        
        
        mis = np.vstack([electrodes, mis])
        mis = mis.T
        
        normalized_mis = np.vstack([electrodes, normalized_mis])
        normalized_mis = normalized_mis.T
        
        return mis, normalized_mis
    
    return mis, normalized_mis
    
def min_max_normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

import seaborn as sns
import matplotlib.pyplot as plt
def plot_heatmap(data, yticklabels=None):
    """
    Plots a heatmap for an Nx1 array (vertical orientation).

    Parameters:
        data (numpy.ndarray): Nx1 array for visualization.
        yticklabels (list, optional): Labels for the y-axis. If None, indices will be used.
    """
    if yticklabels is None:
        yticklabels = list(range(data.shape[0]))  # Automatically generate indices as labels
    
    if len(data.shape) == 1:
        data = np.reshape(data, (-1, 1))
    
    data = np.array(data, dtype=float)
    
    plt.figure(figsize=(2, 10))
    sns.heatmap(
        data, 
        cmap='Blues',
        annot=False,
        linewidths=0.5, 
        xticklabels=False, 
        yticklabels=yticklabels
    )
    plt.title("Vertical Heatmap of Nx1 Array")
    plt.show()

def downsample_mean(data, factor=200):
    channels, points = data.shape
    truncated_length = points - (points % factor)  # 确保整除
    data_trimmed = data[:, :truncated_length]  # 截断到可整除的长度
    data_downsampled = data_trimmed.reshape(channels, -1, factor).mean(axis=2)  # 每 factor 组取平均值
    return data_downsampled

def downsample_decimate(data, factor=200):
    return scipy.signal.decimate(data, factor, axis=1, ftype='fir', zero_phase=True)

def upsample(data, factor=200):
    new_length = len(data) * factor
    data_upsampled = scipy.signal.resample(data, new_length)
    return data_upsampled
    

# labels = utils.read_labels()
# eeg_samples = utils.load_dataset('SEED', subject=1, experiment=1)

# # Downsampling
# eeg_samples_downsampled = downsample_mean(eeg_samples, factor=200)
# mis_downsampling, mis_normalized_downsampling = compute_mis(eeg_samples_downsampled, labels)

# labels = utils.read_labels()
# eeg_samples = utils.load_dataset('SEED', subject=1, experiment=1)

# # Upsampling
# labels_upsampled = upsample(labels, factor=200)
# eeg_samples = eeg_samples[:, :len(labels_upsampled)]

# mis_upsampled, mis_normalized_upsampled = compute_mis(eeg_samples, labels_upsampled)


if __name__ == "__main__":
    # labels upsampling    
    labels = utils.read_labels('SEED')
    labels_upsampled = upsample(labels, 200)
    
    # compute mis_mean
    mis, mis_normalized = [], []
    subject_range, experiment_range = range(1,2), range(1,4)
    for subject in subject_range:
        for experiment in experiment_range:
            eeg_samples = utils.load_dataset('SEED', subject=subject, experiment=experiment)
            eeg_samples = eeg_samples[:, :len(labels_upsampled)]
            
            mis_temp, mis_sampled_temp = compute_mis(eeg_samples, labels_upsampled)
            
            mis.append(mis_temp)
            mis_normalized.append(mis_normalized)
    
    mis = np.array(mis)
    mis_flt = mis[:,:,1]
    
    mis_mean = np.array(np.mean(mis_flt, axis=0), dtype=float)
    
    # arrangement
    distribution = utils.get_distribution()
    electrodes = distribution['channel']
    mis_mean_ = pd.DataFrame({'electrodes':electrodes, 'mi_mean':mis_mean})
    
    # plot heatmap
    plot_heatmap(mis_mean, electrodes)
    plot_heatmap(np.log(mis_mean), electrodes)
    
    # get ascending indices
    mis_mean__ = mis_mean_.sort_values('mi_mean', ascending=False)
    plot_heatmap(mis_mean__['mi_mean'], electrodes)