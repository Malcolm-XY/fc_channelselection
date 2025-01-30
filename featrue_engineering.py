# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 13:42:06 2025

@author: usouu
"""
import os
import numpy
import h5py
import scipy

import matplotlib.pyplot as plt

import utils

import mne
import mne_connectivity
from mne_connectivity import spectral_connectivity_time
from mne_connectivity.viz import plot_connectivity_circle

# %% read original eeg
def read_eeg(experiment, transpose=False, stack=True, freq=200, channelmark=True):
    path_current = os.getcwd()
    path_parent = os.path.dirname(path_current)
    path_folder = os.path.join(path_parent, 'data', 'SEED', 'original eeg', 'Preprocessed_EEG')
    path_file = os.path.join(path_folder, experiment+'.mat')
    
    eeg_mat = read_mat(path_file, transpose=transpose, stack=stack)
    
    if channelmark:
        ch_names=list(utils.get_electrodes())
        info = mne.create_info(ch_names, sfreq=freq, ch_types='eeg')
    else:
        info = mne.create_info(ch_names=['Ch' + str(i) for i in range(eeg_mat.shape[0])], sfreq=freq, ch_types='eeg')
    mneeeg = mne.io.RawArray(eeg_mat, info)
    
    return eeg_mat, mneeeg

def read_mat(path_file, transpose=False, stack=False):
    if not os.path.exists(path_file):
        raise FileNotFoundError(f"File not found: {path_file}")

    try:
        with h5py.File(path_file, 'r') as f:
            print("HDF5 format detected.")
            mat_data = {
                key: numpy.array(f[key]).T if isinstance(f[key], h5py.Dataset) and len(f[key].shape) >= 2 else numpy.array(f[key])
                for key in f.keys()
            }

    except OSError:
        print("Not an HDF5 format.")
        mat_data = scipy.io.loadmat(path_file)
        if transpose:
            mat_data = {
                key: value.T if isinstance(value, numpy.ndarray) and value.ndim >= 2 else value
                for key, value in mat_data.items() if not key.startswith('__')
            }
            if stack:
                mat_data = numpy.vstack([mat_data[key] for key in sorted(mat_data.keys())])
        else:
            mat_data = {
                key: value if isinstance(value, numpy.ndarray) and value.ndim >= 2 else value
                for key, value in mat_data.items() if not key.startswith('__')
            }
            if stack:
                mat_data = numpy.hstack([mat_data[key] for key in sorted(mat_data.keys())])

    return mat_data

# %% filter eeg
def read_filtered_eegdata(experiment, freq_band="Joint"):
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
    path_folder = os.path.join(path_parent, 'data', 'SEED', 'original eeg', 'Filtered_EEG')

    try:
        if freq_band in ["alpha", "beta", "gamma", "delta", "theta"]:
            path_file = os.path.join(path_folder, f"{experiment}_{freq_band.capitalize()}_eeg.fif")
            filtered_eeg = mne.io.read_raw_fif(path_file, preload=True)
            return filtered_eeg

        elif freq_band.lower() == "joint":
            filtered_eeg = {}
            for band in ["Alpha", "Beta", "Gamma", "Delta", "Theta"]:
                path_file = os.path.join(path_folder, f"{experiment}_{band}_eeg.fif")
                filtered_eeg[band.lower()] = mne.io.read_raw_fif(path_file, preload=True)
            return filtered_eeg

        else:
            raise ValueError(f"Invalid frequency band: {freq_band}. Choose from 'alpha', 'beta', 'gamma', 'delta', 'theta', or 'joint'.")

    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found for experiment '{experiment}' and frequency band '{freq_band}'. Check the path and file existence.")

def filter_eeg(eeg, freq=200, verbose=False):
    info = mne.create_info(ch_names=['Ch' + str(i) for i in range(eeg.shape[0])], sfreq=freq, ch_types='eeg')
    mneeeg = mne.io.RawArray(eeg, info)
    
    freq_bands = {
        "Delta": (0.5, 4),
        "Theta": (4, 8),
        "Alpha": (8, 13),
        "Beta": (13, 30),
        "Gamma": (30, 99),
    }
    
    band_filtered_eeg = {}

    for band, (low_freq, high_freq) in freq_bands.items():
        filtered_eeg = mneeeg.copy().filter(l_freq=low_freq, h_freq=high_freq, method="fir", phase="zero-double")
        band_filtered_eeg[band] = filtered_eeg
        if verbose:
            print(f"{band} band filtered: {low_freq}–{high_freq} Hz")

    return band_filtered_eeg

def filter_eeg_and_save(experiment, verbose=True):
    eeg = read_eeg(experiment)  # 假设这个函数已经定义
    
    path_current = os.getcwd()
    path_parent = os.path.dirname(path_current)
    path_folder = os.path.join(path_parent, 'data', 'SEED', 'original eeg', 'Filtered_EEG')
    os.makedirs(path_folder, exist_ok=True)  # 确保目标文件夹存在
    
    # 调用 filter_eeg 函数
    filtered_eeg_dict = filter_eeg(eeg, verbose=verbose)
    
    # 保存每个频段的数据
    for band, filtered_eeg in filtered_eeg_dict.items():
        path_file = os.path.join(path_folder, f"{experiment}_{band}.fif")
        filtered_eeg.save(path_file, overwrite=True)
        if verbose:
            print(f"Saved {band} band filtered EEG to {path_file}")
    
    return filtered_eeg_dict
    
def filter_eeg_and_save_circle():
    for subject in range(1,16):
        for experiment in range(1,4):
            filter_eeg_and_save(f"sub{subject}ex{experiment}")
            
# %% feature engineering
def compute_temporal_connectivity(experiment, method, freq_band, 
                                  window=1, overlap=0, verbose=True, visualization=True):
    filtered_eeg = read_filtered_eegdata(experiment, freq_band=freq_band)
    eeg = filtered_eeg.get_data()

# %% spectral connectivity
def compute_spectral_connectivity(experiment, method, freq_band, 
                                  window=1, overlap=0, freq_density=1, verbose=True):
    # %% asign eeg
    _, eeg = read_eeg(experiment)
    epochs = mne.make_fixed_length_epochs(eeg, duration=window, overlap=0)
    
    try:
        _, eeg = read_eeg(experiment)
    except FileNotFoundError as e:
        raise ValueError(f"Experiment file not found: {experiment}") from e
    
    # parameters
    fmin, fmax = 2, 50
    freqs = numpy.linspace(fmin, fmax, int((fmax-fmin)/freq_density))
    
    if freq_band:
        freq_band_map = {
            "alpha": (8, 13),
            "beta": (13, 30),
            "gamma": (30, 50),
            "theta": (2, 4),
            "delta": (4, 8),
        }
        if freq_band in freq_band_map:
            fmin, fmax = freq_band_map[freq_band]
        else:
            raise ValueError(f"Invalid freq_band '{freq_band}'. Choose from {list(freq_band_map.keys())}.")
  
    # %% compute spectral connectivity
    con = spectral_connectivity_time(epochs, freqs=freqs, method=method,
        fmin=fmin, fmax=fmax, faverage=True, verbose=True)
    
    conn_matrix = con.get_data()
    conn_matrix = numpy.mean(conn_matrix, axis=(2))
    n_channels = int(numpy.sqrt(conn_matrix.shape[1]))
    conn_matrix = conn_matrix.reshape((-1, n_channels, n_channels))
    
    # %% visualization; matplot
    if verbose:
        # get labels
        labels = eeg.ch_names
        
        # get conn matrix
        conn_matrix_avg = numpy.mean(conn_matrix, axis=(0))
        
        # plot heatmap
        plt.figure(figsize=(18, 15))
        plt.imshow(conn_matrix_avg, cmap='viridis', origin='lower')
        plt.colorbar(label='Connectivity Strength')
        plt.title('Functional Connectivity Matrix')
        plt.xlabel('Channels')
        plt.ylabel('Channels')

        plt.xticks(range(n_channels), labels, rotation=90)
        plt.yticks(range(n_channels), labels)

        plt.show()
        
        # %% visualization; mne
        plot_connectivity_circle(conn_matrix_avg, node_names=labels, 
                                 n_lines=62, title= 'Top Functional Connections',
                                 facecolor='white', textcolor='black')

    return conn_matrix

# %% usage
if __name__ == "__main__":
    eeg_mat, eeg = read_eeg("sub1ex1")
    epochs = mne.make_fixed_length_epochs(eeg, duration=1, overlap=0)
    epoched = epochs.get_data()
    
    # filtered_eeg = filter_eeg(eeg)
    filtered_eeg = read_filtered_eegdata("sub1ex1", freq_band="gamma")
    
    eeg = filtered_eeg.get_data()
    # pli = compute_spectral_connectivity("sub1ex1", "pli", "gamma", window=3000)