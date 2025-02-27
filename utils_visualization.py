# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 18:02:23 2025

@author: usouu
"""

import os
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

# %% Visualization
def draw_heatmap_1d(data, yticklabels=None):
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
    # define path
    path_current = os.getcwd()
    
    path_ranking = os.path.join(path_current, 'Distribution', 'electrodes_ranking.txt')
    # read txt; electrodes ranking
    ranking = pd.read_csv(path_ranking, sep='\t')
    
    return ranking

# # %% Example Usage
# if __name__ == '__main__':
