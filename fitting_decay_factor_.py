# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 15:15:50 2025

@author: usouu
"""
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import boxcox
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import fc_computer
from channel_selection_weight_mapping import draw_weight_mapping

def prepare_target_and_inputs(
    feature='PCC',
    ranking_method='label_driven_mi_origin',
    distance_method='euclidean',
    normalize_method='minmax', 
    transform_method='boxcox',    
    mean_align_method='match_mean',
):
    def apply_transform(x):
        x = x + 1e-6  # avoid log(0) or boxcox(0)
        if transform_method == 'boxcox':
            x, _ = boxcox(x)
        elif transform_method == 'sqrt':
            x = np.sqrt(x)
        elif transform_method == 'log':
            x = np.log(x)
        elif transform_method == 'none':
            pass
        else:
            raise ValueError(f"Unsupported transform_method: {transform_method}")
        return x

    weight_mean, index = draw_weight_mapping(ranking_method=ranking_method)

    if normalize_method == 'minmax':
        r_target = MinMaxScaler().fit_transform(weight_mean.to_numpy().reshape(-1, 1)).flatten()
    elif normalize_method == 'zscore':
        r_target = StandardScaler().fit_transform(weight_mean.to_numpy().reshape(-1, 1)).flatten()
    elif normalize_method == 'none':
        r_target = weight_mean.to_numpy()
    else:
        raise ValueError(f"Unsupported normalize_method: {normalize_method}")

    r_target = apply_transform(r_target)
    r_target_mean = np.mean(r_target)

    _, distance_matrix = fc_computer.compute_distance_matrix('seed', method=distance_method)
    distance_matrix = fc_computer.normalize_matrix(distance_matrix)

    _, _, _, global_joint_average = fc_computer.load_global_averages(feature=feature)
    connectivity_matrix = fc_computer.normalize_matrix(global_joint_average)

    def preprocessing_fn(r_fitting):
        if normalize_method == 'minmax':
            r_fitting = MinMaxScaler().fit_transform(r_fitting.reshape(-1, 1)).flatten()
        elif normalize_method == 'zscore':
            r_fitting = StandardScaler().fit_transform(r_fitting.reshape(-1, 1)).flatten()
        elif normalize_method == 'none':
            r_fitting = r_fitting.copy()
        else:
            raise ValueError(f"Unsupported normalize_method: {normalize_method}")

        r_fitting = apply_transform(r_fitting)

        if mean_align_method == 'match_mean':
            delta = r_target_mean - np.mean(r_fitting)
            r_fitting += delta

        return r_fitting

    return r_target, r_target_mean, distance_matrix, connectivity_matrix, preprocessing_fn

def compute_r_fitting(method, params_dict):
    factor_matrix = fc_computer.compute_volume_conduction_factors(
        distance_matrix, method=method, params=params_dict
    )
    factor_matrix = fc_computer.normalize_matrix(factor_matrix)
    differ_PCC_DM = connectivity_matrix - factor_matrix
    differ_PCC_DM = fc_computer.normalize_matrix(differ_PCC_DM)
    r_fitting = np.mean(differ_PCC_DM, axis=0)
    r_fitting = fc_computer.normalize_matrix(r_fitting)
    r_fitting = preprocessing_fn(r_fitting)
    return r_fitting

def optimize_and_store(name, loss_fn, x0, bounds, param_keys):
    res = minimize(loss_fn, x0=x0, bounds=bounds)
    params = dict(zip(param_keys, res.x))
    results[name] = {'params': params, 'loss': res.fun}
    fittings[name] = compute_r_fitting(name, params)

def loss_exponential(params):
    return np.mean((compute_r_fitting('exponential', {'sigma': params[0]}) - r_target) ** 2)

def loss_gaussian(params):
    return np.mean((compute_r_fitting('gaussian', {'sigma': params[0]}) - r_target) ** 2)

def loss_inverse(params):
    return np.mean((compute_r_fitting('inverse', {'sigma': params[0], 'alpha': params[1]}) - r_target) ** 2)

def loss_powerlaw(params):
    return np.mean((compute_r_fitting('powerlaw', {'alpha': params[0]}) - r_target) ** 2)

def loss_rational_quadratic(params):
    return np.mean((compute_r_fitting('rational_quadratic', {'sigma': params[0], 'alpha': params[1]}) - r_target) ** 2)

def loss_generalized_gaussian(params):
    return np.mean((compute_r_fitting('generalized_gaussian', {'sigma': params[0], 'beta': params[1]}) - r_target) ** 2)

def loss_sigmoid(params):
    return np.mean((compute_r_fitting('sigmoid', {'mu': params[0], 'beta': params[1]}) - r_target) ** 2)

if __name__ == '__main__':
    results = {}
    fittings = {}

    r_target, r_target_mean, distance_matrix, connectivity_matrix, preprocessing_fn = prepare_target_and_inputs(
        feature='PCC',
        ranking_method='label_driven_mi_origin',
        distance_method='stereo',
        transform_method='boxcox',
    )

    optimize_and_store('exponential', loss_exponential, x0=[2.0], bounds=[(0.1, 20.0)], param_keys=['sigma'])
    optimize_and_store('gaussian', loss_gaussian, x0=[2.0], bounds=[(0.1, 20.0)], param_keys=['sigma'])
    optimize_and_store('inverse', loss_inverse, x0=[2.0, 2.0], bounds=[(0.1, 20.0), (0.1, 5.0)], param_keys=['sigma', 'alpha'])
    optimize_and_store('powerlaw', loss_powerlaw, x0=[2.0], bounds=[(0.1, 10.0)], param_keys=['alpha'])
    optimize_and_store('rational_quadratic', loss_rational_quadratic, x0=[2.0, 1.0], bounds=[(0.1, 20.0), (0.1, 10.0)], param_keys=['sigma', 'alpha'])
    optimize_and_store('generalized_gaussian', loss_generalized_gaussian, x0=[2.0, 1.0], bounds=[(0.1, 20.0), (0.1, 5.0)], param_keys=['sigma', 'beta'])
    optimize_and_store('sigmoid', loss_sigmoid, x0=[2.0, 1.0], bounds=[(0.1, 10.0), (0.1, 5.0)], param_keys=['mu', 'beta'])

    print("=== Fitting Results of All Models (Minimum MSE) ===")
    for method, result in results.items():
        print(f"[{method.upper()}] Best Parameters: {result['params']}, Minimum MSE: {result['loss']:.6f}")

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()

    for idx, (method, r_fitting) in enumerate(fittings.items()):
        ax = axes[idx]
        ax.plot(r_target, label='r_target', linestyle='--', marker='o')
        ax.plot(r_fitting, label=f'r_fitting ({method})', marker='x')
        ax.set_title(f"{method.upper()} - MSE: {results[method]['loss']:.4e}")
        ax.set_xlabel("Channel Index")
        ax.set_ylabel("Importance (normalized)")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.suptitle("Fitting Comparison of Channel Importance across Models", fontsize=18, y=1.02)
    plt.show()

    plt.figure(figsize=(12, 6))
    bar_width = 0.1
    x = np.arange(len(r_target))
    
    # Heatmap visualization
    heatmap_data = np.vstack([r_target] + [fittings[method] for method in fittings.keys()])
    heatmap_labels = ['target'] + list(fittings.keys())
    
    plt.figure(figsize=(14, 6))
    sns.heatmap(heatmap_data, cmap='viridis', cbar=True, xticklabels=False, yticklabels=heatmap_labels, linewidths=0.5, linecolor='gray')
    plt.title("Heatmap of r_target and All r_fitting Vectors")
    plt.xlabel("Channel Index")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.show()
