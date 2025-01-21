# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 13:42:06 2025

@author: usouu
"""
import os
import numpy
import h5py
import scipy

import mne

def read_eeg(subject, experiment, transpose=False, stack=True):
    path_current = os.getcwd()
    path_parent = os.path.dirname(path_current)
    path_folder = os.path.join(path_parent, 'data', 'SEED', 'original eeg', 'Preprocessed_EEG')
    path_file = os.path.join(path_folder, subject+experiment+'.mat')
    
    eeg_mat = read_mat(path_file, transpose=transpose, stack=stack)
    return eeg_mat

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

def load_eegdata():
    path_current = os.getcwd()
    path_parent = os.path.dirname(path_current)
    path_folder = os.path.join(path_parent, 'data', 'SEED', 'original eeg', 'Preprocessed_EEG')

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

def filter_eeg_and_save(subject, experiment, verbose=True):
    eeg = read_eeg(subject, experiment)  # 假设这个函数已经定义
    
    path_current = os.getcwd()
    path_parent = os.path.dirname(path_current)
    path_folder = os.path.join(path_parent, 'data', 'SEED', 'original eeg', 'Filtered_EEG')
    os.makedirs(path_folder, exist_ok=True)  # 确保目标文件夹存在
    
    # 调用 filter_eeg 函数
    filtered_eeg_dict = filter_eeg(eeg, verbose=verbose)
    
    # 保存每个频段的数据
    for band, filtered_eeg in filtered_eeg_dict.items():
        path_file = os.path.join(path_folder, f"{subject}{experiment}_{band}.fif")
        filtered_eeg.save(path_file, overwrite=True)
        if verbose:
            print(f"Saved {band} band filtered EEG to {path_file}")
    
    return filtered_eeg_dict
    
band_filtered_eeg = filter_eeg_and_save('sub1','ex1')