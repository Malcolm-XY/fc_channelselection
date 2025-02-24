# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 14:11:18 2025

@author: 18307
"""

import os
import pickle
import pandas as pd

def read_pkl(path_file, method='pd'):
    if method == 'pd':
        data = pd.read_pickle(path_file)
    
    elif method == 'pkl':
        with open(path_file, 'rb') as file:
            data = pickle.load(file)
        
    return data

def read_functional_connectivity_pkl(identifier, feature, method='pkl'):
    path_current = os.getcwd()
    path_parent = os.path.dirname(path_current)
    path_parent_parent = os.path.dirname(path_parent)
    path_fc_features = os.path.join(path_parent_parent, 'Research_Data', 'SEED', 'functional connectivity')
    
    if method == 'pkl':
        if feature == 'pcc':
            path_data = os.path.join()
            data = read_pkl()
    

# %% Example Usage
if __name__ == '__main__':
    path_current = os.getcwd()
    path_parent = os.path.dirname(path_current)
    path_parent_parent = os.path.dirname(path_parent)
    path_fc_features = os.path.join(path_parent_parent, 'Research_Data', 'SEED', 'functional connectivity')
    path_mi_features = os.path.join(path_fc_features, 'mi_pkl')
    
    example_path = os.path.join(path_mi_features, 'sub1ex1_alpha.pkl')
    
    data = read_pkl(example_path)