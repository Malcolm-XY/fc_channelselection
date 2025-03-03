# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 00:17:25 2025

@author: 18307
"""

import os
import h5py
import scipy
import numpy as np
import pandas as pd

# %% Basic File Reading Functions
def read_txt(path_file):
    """
    Reads a text file and returns its content as a Pandas DataFrame.
    
    Parameters:
    - path_file (str): Path to the text file.
    
    Returns:
    - pd.DataFrame: DataFrame containing the parsed text data.
    """
    return pd.read_csv(path_file, sep=r'\s+', engine='python')

def read_hdf5(path_file):
    """
    Reads an HDF5 file and returns its contents as a dictionary.
    
    Parameters:
    - path_file (str): Path to the HDF5 file.
    
    Returns:
    - dict: Parsed data from the HDF5 file.
    
    Raises:
    - FileNotFoundError: If the file does not exist.
    - TypeError: If the file is not a valid HDF5 format.
    """
    if not os.path.exists(path_file):
        raise FileNotFoundError(f"File not found: {path_file}")

    try:
        with h5py.File(path_file, 'r') as f:
            return {key: simplify_mat_structure(f[key]) for key in f.keys()}
    except OSError:
        raise TypeError(f"File '{path_file}' is not in HDF5 format.")

def read_mat(path_file, simplify=True):
    """
    Reads a MATLAB .mat file, supporting both HDF5 and older formats.
    
    Parameters:
    - path_file (str): Path to the .mat file.
    - simplify (bool): Whether to simplify the data structure (default: True).
    
    Returns:
    - dict: Parsed MATLAB file data.
    
    Raises:
    - FileNotFoundError: If the file does not exist.
    - TypeError: If the file format is invalid.
    """
    if not os.path.exists(path_file):
        raise FileNotFoundError(f"File not found: {path_file}")
    
    try:
        # Attempt to read as HDF5 format
        with h5py.File(path_file, 'r') as f:
            return {key: simplify_mat_structure(f[key]) for key in f.keys()} if simplify else f
    except OSError:
        try:
            # Read as non-HDF5 .mat file
            mat_data = scipy.io.loadmat(path_file, squeeze_me=simplify, struct_as_record=not simplify)
            return {key: simplify_mat_structure(value) for key, value in mat_data.items() if not key.startswith('_')} if simplify else mat_data
        except Exception as e:
            raise TypeError(f"Failed to read '{path_file}': {e}")

def simplify_mat_structure(data):
    """
    Recursively processes and simplifies MATLAB data structures.
    
    Converts:
    - HDF5 datasets to NumPy arrays or scalars.
    - HDF5 groups to Python dictionaries.
    - MATLAB structs to Python dictionaries.
    - Cell arrays to Python lists.
    - NumPy arrays are squeezed to remove unnecessary dimensions.
    
    Parameters:
    - data: Input data (HDF5, MATLAB struct, NumPy array, etc.).
    
    Returns:
    - Simplified Python data structure.
    """
    if isinstance(data, h5py.Dataset):
        return data[()]
    elif isinstance(data, h5py.Group):
        return {key: simplify_mat_structure(data[key]) for key in data.keys()}
    elif isinstance(data, scipy.io.matlab.mat_struct):
        return {field: simplify_mat_structure(getattr(data, field)) for field in data._fieldnames}
    elif isinstance(data, np.ndarray):
        if data.dtype == 'object':
            return [simplify_mat_structure(item) for item in data]
        return np.squeeze(data)
    return data

# %% Read Labels Functions
def read_labels(dataset):
    """
    Reads emotion labels for a specified dataset.
    
    Parameters:
    - dataset (str): The dataset name (e.g., 'SEED', 'DREAMER').
    
    Returns:
    - pd.DataFrame: DataFrame containing label data.
    
    Raises:
    - ValueError: If the dataset is not supported.
    """
    path_parent_parent = os.path.dirname(os.path.dirname(os.getcwd()))
    if dataset.lower() == 'seed':
        path_labels = os.path.join(path_parent_parent, 'Research_Data', 'SEED', 'labels', 'labels_seed.txt')
    elif dataset.lower() == 'dreamer':
        path_labels = os.path.join(path_parent_parent, 'Research_Data', 'DREAMER', 'labels', 'labels_dreamer.txt')
    else:
        raise ValueError('Currently only support SEED and DREAMER')
    return read_txt(path_labels)

# %% Read Feature Functions
def read_cfs(dataset, identifier, feature, band='joint'):
    """
    Reads channel feature data (CFS) from an HDF5 file.
    
    Parameters:
    - dataset (str): Dataset name (e.g., 'SEED').
    - identifier (str): Subject/Experiment identifier.
    - feature (str): Feature type.
    - band (str): Frequency band (default: 'joint').
    
    Returns:
    - dict: Parsed CFS data.
    """
    dataset, identifier, feature, band = dataset.upper(), identifier.lower(), feature.lower(), band.lower()
    path_parent_parent = os.path.dirname(os.path.dirname(os.getcwd()))
    path_file = os.path.join(path_parent_parent, 'Research_Data', dataset, 'channel features', f'{feature}_h5', f"{identifier}.h5")
    cfs_temp = read_hdf5(path_file)
    return cfs_temp if band == 'joint' else cfs_temp.get(band, {})

def read_fcs(dataset, identifier, feature, band='joint'):
    """
    Reads functional connectivity data (FCS) from an HDF5 file.
    
    Parameters:
    - dataset (str): Dataset name (e.g., 'SEED').
    - identifier (str): Subject/Experiment identifier.
    - feature (str): Feature type.
    - band (str): Frequency band (default: 'joint').
    
    Returns:
    - dict: Parsed FCS data.
    """
    dataset, identifier, feature, band = dataset.upper(), identifier.lower(), feature.lower(), band.lower()
    path_parent_parent = os.path.dirname(os.path.dirname(os.getcwd()))
    path_file = os.path.join(path_parent_parent, 'Research_Data', dataset, 'functional connectivity', f'{feature}_h5', f"{identifier}.h5")
    fcs_temp = read_hdf5(path_file)
    return fcs_temp if band == 'joint' else fcs_temp.get(band, {})

# %% Example Usage
if __name__ == "__main__":
    # %% cfs
    dataset, experiment_sample, feature_sample, freq_sample = 'seed', 'sub1ex1', 'de_LDS', 'joint'
    seed_cfs_sample = read_cfs(dataset, experiment_sample, feature_sample, freq_sample)
    
    # %% fcs
    dataset, experiment_sample, feature_sample, freq_sample = 'seed', 'sub1ex1', 'pcc', 'joint'
    seed_fcs_sample_seed = read_fcs(dataset, experiment_sample, feature_sample, freq_sample)
    
    dataset, experiment_sample = 'dreamer', 'sub1'
    seed_fcs_sample_dreamer = read_fcs(dataset, experiment_sample, feature_sample, freq_sample)
    
    # %% read labels
    labels_seed_ = read_labels('seed')
    
    labels_dreamer_ = read_labels('dreamer')
