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

def read_eeg_mat(path_file, transpose=False, stack=False):
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

    return mat_data

def load_eegdata():
    path_current = os.getcwd()
    path_parent = os.path.dirname(path_current)
    path_folder = os.path.join(path_parent, 'data', 'SEED', 'original eeg', 'Preprocessed_EEG')

path_current = os.getcwd()
path_parent = os.path.dirname(path_current)
path_folder = os.path.join(path_parent, 'data', 'SEED', 'original eeg', 'Preprocessed_EEG')
path_file = os.path.join(path_folder, '1_20131027.mat')
    
eeg_mat = read_eeg_mat(path_file, transpose=True, stack=True)

raw = mne.io.read_raw(path_file, preload=True)

print(f"数据加载成功，采样率: {raw.info['sfreq']} Hz, 通道数: {len(raw.ch_names)}")