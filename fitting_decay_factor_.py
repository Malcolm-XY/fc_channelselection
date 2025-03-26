# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 17:30:25 2025

@author: usouu
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import boxcox
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from functools import partial

import featrue_engineering
import feature_transformation
import weight_map_drawer

def apply_transform(x, method):
    x = x + 1e-6  # avoid log(0) or boxcox(0)
    if method == 'boxcox':
        x, _ = boxcox(x)
    elif method == 'sqrt':
        x = np.sqrt(x)
    elif method == 'log':
        x = np.log(x)
    elif method == 'none':
        pass
    else:
        raise ValueError(f"Unsupported transform_method: {method}")
    return x

# def normalize_data(x, method):
#     if method == 'minmax':
#         return MinMaxScaler().fit_transform(x.reshape(-1, 1)).flatten()
#     elif method == 'zscore':
#         return StandardScaler().fit_transform(x.reshape(-1, 1)).flatten()
#     elif method == 'none':
#         return x.copy()
#     else:
#         raise ValueError(f"Unsupported normalize_method: {method}")

def preprocessing_r_target(r, normalize_method, transform_method):
    r = featrue_engineering.normalize_matrix(r, normalize_method)
    r = apply_transform(r, transform_method)

    return r

def preprocessing_r_fitting(r, normalize_method, transform_method, r_target, mean_align_method):
    r = featrue_engineering.normalize_matrix(r, normalize_method)
    r = apply_transform(r, transform_method)

    if mean_align_method == 'match_mean':
        delta = np.mean(r_target) - np.mean(r)
        r += delta

    return r

def prepare_target_and_inputs(
    feature='PCC',
    ranking_method='label_driven_mi_origin',
    distance_method='euclidean',
    normalize_method='minmax',
    transform_method='boxcox',
    mean_align_method='match_mean',
):
    weight_mean, index = draw_weight_mapping(ranking_method=ranking_method)   
    r_target = preprocessing_r_target(weight_mean.to_numpy(), normalize_method, transform_method)

    _, distance_matrix = feature_transformation.compute_distance_matrix('seed', method=distance_method)
    distance_matrix = featrue_engineering.normalize_matrix(distance_matrix)

    _, _, _, global_joint_average = feature_transformation.load_global_averages(feature=feature)
    connectivity_matrix = featrue_engineering.normalize_matrix(global_joint_average)

    preprocessing_fn = partial(
        preprocessing_r_fitting,
        normalize_method=normalize_method,
        transform_method=transform_method,
        r_target=r_target,
        mean_align_method=mean_align_method
    )

    return r_target, distance_matrix, connectivity_matrix, preprocessing_fn

def compute_r_fitting(method, params_dict, distance_matrix, connectivity_matrix, preprocessing_fn):
    factor_matrix = feature_transformation.compute_volume_conduction_factors(distance_matrix, method=method, params=params_dict)
    factor_matrix = featrue_engineering.normalize_matrix(factor_matrix)
    differ_PCC_DM = featrue_engineering.normalize_matrix(connectivity_matrix - factor_matrix)
    r_fitting = np.mean(differ_PCC_DM, axis=0)
    r_fitting = featrue_engineering.normalize_matrix(r_fitting)
    return preprocessing_fn(r_fitting)

def optimize_and_store(name, loss_fn, x0, bounds, param_keys, distance_matrix, connectivity_matrix, preprocessing_fn):
    res = minimize(loss_fn, x0=x0, bounds=bounds)
    params = dict(zip(param_keys, res.x))
    results[name] = {'params': params, 'loss': res.fun}
    fittings[name] = compute_r_fitting(name, params, distance_matrix, connectivity_matrix, preprocessing_fn)

def loss_fn_template(method_name, param_dict_fn, r_target, distance_matrix, connectivity_matrix, preprocessing_fn):
    def loss(params):
        return np.mean((compute_r_fitting(method_name, param_dict_fn(params), distance_matrix, connectivity_matrix, preprocessing_fn) - r_target) ** 2)
    return loss

if __name__ == '__main__':
    results = {}
    fittings = {}

    r_target, distance_matrix, connectivity_matrix, preprocessing_fn = prepare_target_and_inputs(
        feature='PCC',
        ranking_method='label_driven_mi_origin',
        distance_method='stereo',
        transform_method='boxcox',
    )

    optimize_and_store('exponential', loss_fn_template('exponential', lambda p: {'sigma': p[0]}, r_target, distance_matrix, connectivity_matrix, preprocessing_fn), [2.0], [(0.1, 20.0)], ['sigma'], distance_matrix, connectivity_matrix, preprocessing_fn)
    optimize_and_store('gaussian', loss_fn_template('gaussian', lambda p: {'sigma': p[0]}, r_target, distance_matrix, connectivity_matrix, preprocessing_fn), [2.0], [(0.1, 20.0)], ['sigma'], distance_matrix, connectivity_matrix, preprocessing_fn)
    optimize_and_store('inverse', loss_fn_template('inverse', lambda p: {'sigma': p[0], 'alpha': p[1]}, r_target, distance_matrix, connectivity_matrix, preprocessing_fn), [2.0, 2.0], [(0.1, 20.0), (0.1, 5.0)], ['sigma', 'alpha'], distance_matrix, connectivity_matrix, preprocessing_fn)
    optimize_and_store('powerlaw', loss_fn_template('powerlaw', lambda p: {'alpha': p[0]}, r_target, distance_matrix, connectivity_matrix, preprocessing_fn), [2.0], [(0.1, 10.0)], ['alpha'], distance_matrix, connectivity_matrix, preprocessing_fn)
    optimize_and_store('rational_quadratic', loss_fn_template('rational_quadratic', lambda p: {'sigma': p[0], 'alpha': p[1]}, r_target, distance_matrix, connectivity_matrix, preprocessing_fn), [2.0, 1.0], [(0.1, 20.0), (0.1, 10.0)], ['sigma', 'alpha'], distance_matrix, connectivity_matrix, preprocessing_fn)
    optimize_and_store('generalized_gaussian', loss_fn_template('generalized_gaussian', lambda p: {'sigma': p[0], 'beta': p[1]}, r_target, distance_matrix, connectivity_matrix, preprocessing_fn), [2.0, 1.0], [(0.1, 20.0), (0.1, 5.0)], ['sigma', 'beta'], distance_matrix, connectivity_matrix, preprocessing_fn)
    optimize_and_store('sigmoid', loss_fn_template('sigmoid', lambda p: {'mu': p[0], 'beta': p[1]}, r_target, distance_matrix, connectivity_matrix, preprocessing_fn), [2.0, 1.0], [(0.1, 10.0), (0.1, 5.0)], ['mu', 'beta'], distance_matrix, connectivity_matrix, preprocessing_fn)

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

    heatmap_data = np.vstack([r_target] + [fittings[method] for method in fittings.keys()])
    heatmap_labels = ['target'] + list(fittings.keys())

    plt.figure(figsize=(14, 6))
    sns.heatmap(heatmap_data, cmap='viridis', cbar=True, xticklabels=False, yticklabels=heatmap_labels, linewidths=0.5, linecolor='gray')
    plt.title("Heatmap of r_target and All r_fitting Vectors")
    plt.xlabel("Channel Index")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.show()
    
    # %% Validation
    from utils import utils_feature_loading
    electrodes = utils_feature_loading.read_distribution('seed')['channel']    
    # target
    r_target_ = r_target.copy()
    _, strength_ranked, in_original_indices = feature_transformation.rank_and_visualize_fc_strength(r_target_, electrodes)
    feature_transformation.draw_weight_rank_mapping(in_original_indices, strength_ranked['Strength'])
    
    # non-fitted
    _,_,_,r_non_fitted = feature_transformation.load_global_averages(feature='PCC')
    r_non_fitted = np.mean(r_non_fitted, axis=0)
    _, strength_ranked, in_original_indices = feature_transformation.rank_and_visualize_fc_strength(r_non_fitted, electrodes) #, exclude_electrodes=['CB1', 'CB2'])
    feature_transformation.draw_weight_rank_mapping(in_original_indices, strength_ranked['Strength'])
    
    # fitted
    r_fitted_g_gaussian = fittings['generalized_gaussian']
    _, strength_ranked, in_original_indices = feature_transformation.rank_and_visualize_fc_strength(r_fitted_g_gaussian, electrodes) #, exclude_electrodes=['CB1', 'CB2'])
    feature_transformation.draw_weight_rank_mapping(in_original_indices, strength_ranked['Strength'])
