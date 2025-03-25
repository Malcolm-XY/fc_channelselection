# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 23:07:14 2025

@author: 18307
"""

import os
import h5py
import numpy as np
import pandas as pd

from utils import utils_feature_loading
from utils import utils_visualization

def load_global_averages(file_path=None, feature=None):
    """
    读取 HDF5 文件中的 global_alpha_average, global_beta_average, global_gamma_average 和 global_joint_average 数据。

    Args:
        file_path (str, optional): HDF5 文件的完整路径。若为 None，则根据 feature 参数构造路径。
        feature (str, optional): 特征类型，如 'PCC'。仅当 file_path 为 None 时使用。

    Returns:
        tuple: 包含 global_alpha_average, global_beta_average, global_gamma_average 和 global_joint_average 的元组。
    """
    # 如果没有提供文件路径，则根据特征类型构造默认路径
    if file_path is None:
        if feature is None:
            file_path = 'Distribution/fc_global_averages.h5'
        else:
            path_current = os.getcwd()
            file_path = os.path.join(path_current, 'Distribution', 'Distance_Matrices',
                                     feature.upper(), f'fc_global_averages_{feature}.h5')

    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    # 读取数据
    with h5py.File(file_path, 'r') as f:
        global_alpha_average = f['alpha'][:]
        global_beta_average = f['beta'][:]
        global_gamma_average = f['gamma'][:]
        global_joint_average = f['joint'][:]

    return global_alpha_average, global_beta_average, global_gamma_average, global_joint_average

def compute_averaged_fcnetwork(feature, subjects=range(1, 16), experiments=range(1, 4), draw=True, save=False):
    # 初始化存储结果的列表
    cmdata_averages_dict = []

    # 用于累积频段的所有数据
    all_alpha_values = []
    all_beta_values = []
    all_gamma_values = []

    # 遍历 subject 和 experiment
    for subject in subjects:  # 假设 subjects 是整数
        for experiment in experiments:  # 假设 experiments 是整数
            identifier = f"sub{subject}ex{experiment}"
            print(identifier)
            try:
                # 加载数据
                cmdata_alpha = utils_feature_loading.read_fcs(dataset='seed', identifier=identifier, feature=feature,
                                                              band='alpha')
                cmdata_beta = utils_feature_loading.read_fcs(dataset='seed', identifier=identifier, feature=feature,
                                                             band='beta')
                cmdata_gamma = utils_feature_loading.read_fcs(dataset='seed', identifier=identifier, feature=feature,
                                                              band='gamma')
                # 计算平均值
                cmdata_alpha_averaged = np.mean(cmdata_alpha, axis=0)
                cmdata_beta_averaged = np.mean(cmdata_beta, axis=0)
                cmdata_gamma_averaged = np.mean(cmdata_gamma, axis=0)

                # 累积数据
                all_alpha_values.append(cmdata_alpha_averaged)
                all_beta_values.append(cmdata_beta_averaged)
                all_gamma_values.append(cmdata_gamma_averaged)

                # # 可视化
                # draw_projection(cmdata_alpha_averaged)
                # draw_projection(cmdata_beta_averaged)
                # draw_projection(cmdata_gamma_averaged)

                # 合并同 subject 同 experiment 的数据
                cmdata_averages_dict.append({
                    "subject": subject,
                    "experiment": experiment,
                    "averages": {
                        "alpha": cmdata_alpha_averaged,
                        "beta": cmdata_beta_averaged,
                        "gamma": cmdata_gamma_averaged
                    }
                })
            except Exception as e:
                print(f"Error processing sub {subject} ex {experiment}: {e}")

    # 计算整个数据集的全局平均值
    global_alpha_average = np.mean(all_alpha_values, axis=0)
    global_beta_average = np.mean(all_beta_values, axis=0)
    global_gamma_average = np.mean(all_gamma_values, axis=0)
    global_joint_average = np.mean(np.stack([global_alpha_average, global_beta_average, global_gamma_average], axis=0),
                                   axis=0)

    if draw:
        # 输出结果
        utils_visualization.draw_projection(global_alpha_average)
        utils_visualization.draw_projection(global_beta_average)
        utils_visualization.draw_projection(global_gamma_average)
        utils_visualization.draw_projection(global_joint_average)

    if save:
        # 检查和创建 Distribution 文件夹
        output_dir = 'Distribution'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # 保存为 HDF5 文件
        file_path = os.path.join(output_dir, 'fc_global_averages.h5')
        with h5py.File(file_path, 'w') as f:
            f.create_dataset('alpha', data=global_alpha_average)
            f.create_dataset('beta', data=global_beta_average)
            f.create_dataset('gamma', data=global_gamma_average)
            f.create_dataset('joint', data=global_joint_average)

        print(f"Results saved to {file_path}")

    return global_alpha_average, global_beta_average, global_gamma_average, global_joint_average

def normalize_matrix(matrix, method='minmax'):
    """
    对矩阵进行归一化处理。

    Args:
        matrix (numpy.ndarray): 要归一化的矩阵或数组
        method (str, optional): 归一化方法，可选值为'minmax'、'max'、'mean'、'z-score'。默认为'minmax'。
            - 'minmax': (x - min) / (max - min)，将值归一化到[0,1]区间
            - 'max': x / max，将最大值归一化为1
            - 'mean': x / mean，相对于平均值进行归一化
            - 'z-score': (x - mean) / std，标准化为均值0，标准差1

    Returns:
        numpy.ndarray: 归一化后的矩阵

    Raises:
        ValueError: 当提供的归一化方法不受支持时
    """
    # 创建输入矩阵的副本，避免修改原始数据
    normalized = matrix.copy()

    if method == 'minmax':
        # Min-Max归一化：将值归一化到[0,1]区间
        min_val = np.min(normalized)
        max_val = np.max(normalized)
        if max_val > min_val:  # 避免除以零
            normalized = (normalized - min_val) / (max_val - min_val)

    elif method == 'max':
        # 最大值归一化：将值归一化到[0,1]区间，最大值为1
        max_val = np.max(normalized)
        if max_val > 0:  # 避免除以零
            normalized = normalized / max_val

    elif method == 'mean':
        # 均值归一化：相对于平均值进行归一化
        mean_val = np.mean(normalized)
        if mean_val > 0:  # 避免除以零
            normalized = normalized / mean_val

    elif method == 'z-score':
        # Z-score标准化：将均值归一化为0，标准差归一化为1
        mean_val = np.mean(normalized)
        std_val = np.std(normalized)
        if std_val > 0:  # 避免除以零
            normalized = (normalized - mean_val) / std_val

    else:
        raise ValueError(f"不支持的归一化方法: {method}，可选值为'minmax'、'max'、'mean'或'z-score'")

    return normalized

def compute_distance_matrix(dataset, method='euclidean', normalize=False, normalization_method='minmax',
                            stereo_params=None):
    """
    计算电极之间的距离矩阵，支持多种距离计算方法。

    Args:
        dataset (str): 数据集名称，用于读取分布信息。
        method (str, optional): 距离计算方法，可选值为'euclidean'或'stereo'。默认为'euclidean'。
            - 'euclidean': 直接计算3D空间中的欧几里得距离
            - 'stereo': 首先进行立体投影到2D平面，然后计算投影点之间的欧几里得距离
        normalize (bool, optional): 是否对距离矩阵进行归一化。默认为False。
        normalization_method (str, optional): 归一化方法，可选值见normalize_matrix函数。默认为'minmax'。
        stereo_params (dict, optional): 立体投影的参数，仅当method='stereo'时使用。默认为None，此时使用默认参数。
            可包含以下键值对：
            - 'prominence': 投影的突出参数，默认为0.1
            - 'epsilon': 防止除零的小常数，默认为0.01

    Returns:
        tuple: 包含以下元素:
            - channel_names (list): 通道名称列表
            - distance_matrix (numpy.ndarray): 原始或归一化后的距离矩阵
    """
    import numpy as np

    # 读取电极分布信息
    distribution = utils_feature_loading.read_distribution(dataset)
    channel_names = distribution['channel']
    x, y, z = np.array(distribution['x']), np.array(distribution['y']), np.array(distribution['z'])

    # 设置立体投影的默认参数
    default_stereo_params = {
        'prominence': 0.1,
        'epsilon': 0.01
    }

    # 如果提供了stereo_params，更新默认参数
    if stereo_params is not None:
        default_stereo_params.update(stereo_params)

    if method == 'euclidean':
        # 计算3D欧几里得距离
        coords = np.vstack((x, y, z)).T  # 形状 (N, 3)
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        distance_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

    elif method == 'stereo':
        # 执行立体投影
        prominence = default_stereo_params['prominence']
        epsilon = default_stereo_params['epsilon']

        # 归一化z坐标并应用prominence参数
        z_norm = (z - np.min(z)) / (np.max(z) - np.min(z)) - prominence

        # 计算投影坐标
        x_proj = x / (1 - z_norm + epsilon)
        y_proj = y / (1 - z_norm + epsilon)

        # 归一化投影坐标
        x_norm = (x_proj - np.min(x_proj)) / (np.max(x_proj) - np.min(x_proj))
        y_norm = (y_proj - np.min(y_proj)) / (np.max(y_proj) - np.min(y_proj))

        # 将投影后的2D坐标堆叠成矩阵
        proj_coords = np.vstack((x_norm, y_norm)).T  # 形状 (N, 2)

        # 计算投影点之间的2D欧几里得距离
        diff = proj_coords[:, np.newaxis, :] - proj_coords[np.newaxis, :, :]
        distance_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

    else:
        raise ValueError(f"不支持的距离计算方法: {method}，可选值为'euclidean'或'stereo'")

    # 对距离矩阵进行归一化（如果需要）
    if normalize:
        distance_matrix = normalize_matrix(distance_matrix, method=normalization_method)

    return channel_names, distance_matrix

def rank_and_visualize_fc_network(fc_matrix, electrodes, feature_name='feature', ascending=False, draw=True, exclude_electrodes=None):
    """
    对功能连接(FC)网络进行排序并可视化，并支持在排序后屏蔽指定电极。

    Args:
        fc_matrix (numpy.ndarray): 功能连接矩阵，形状为 (n, n) 或 (m, n, n)
        electrodes (list): 电极标签列表
        feature_name (str, optional): 特征名称，用于DataFrame列名。默认为'feature'
        ascending (bool, optional): 排序顺序，True为升序，False为降序。默认为False
        draw (bool, optional): 是否绘制热图。默认为True
        exclude_electrodes (list[str], optional): 要在排序后排除的电极名称列表

    Returns:
        tuple: 包含以下元素:
            - mean_values (numpy.ndarray): 每个节点的平均连接强度（全部电极）
            - df_original (pandas.DataFrame): 包含电极和平均值的原始DataFrame
            - df_ranked (pandas.DataFrame): 排除电极后的排序DataFrame
            - rank_indices (numpy.ndarray): 排除电极后的索引，可用于重新排列原始矩阵
    """
    import numpy as np
    import pandas as pd

    if fc_matrix.ndim == 3:
        mean_values = np.mean(np.mean(fc_matrix, axis=0), axis=0)
    elif fc_matrix.ndim == 2:
        mean_values = np.mean(fc_matrix, axis=0)
    else:
        raise ValueError(f"输入矩阵维度应为2或3，但得到的是{fc_matrix.ndim}")

    if len(electrodes) != len(mean_values):
        raise ValueError(f"电极数量({len(electrodes)})与连接矩阵节点数({len(mean_values)})不匹配")

    # 确保 electrodes 是列表
    electrodes = list(electrodes)

    # 创建原始 DataFrame
    column_name = "mean"
    df_original = pd.DataFrame({'electrodes': electrodes, column_name: mean_values})

    # 排序
    df_ranked = df_original.sort_values(column_name, ascending=ascending).reset_index(drop=True)

    # 在排序后排除指定电极
    if exclude_electrodes is not None:
        df_ranked = df_ranked[~df_ranked['electrodes'].isin(exclude_electrodes)].reset_index(drop=True)

    # 重新获取排序后电极在原始 electrodes 中的索引
    rank_indices = np.array([electrodes.index(e) for e in df_ranked['electrodes']])

    # 绘图
    if draw:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # 排除电极后的均值
            heat_values = df_ranked[column_name].values
            heat_labels = df_ranked['electrodes'].values

            plt.figure(figsize=(10, 2))
            sns.heatmap([heat_values], cmap='viridis',
                        xticklabels=heat_labels, yticklabels=False)
            plt.title(f"sorted {feature_name} average connection strength\n(excluded: {exclude_electrodes})")
            plt.tight_layout()
            plt.show()

        except ImportError:
            print("Error in drawing heatmap: matplotlib or seaborn not installed.")
        except Exception as e:
            print(f"Error in drawing heatmap: {str(e)}")

    return mean_values, df_original, df_ranked, rank_indices

def rank_and_visualize_fc_network_(fc_matrix, electrodes, feature_name='feature', ascending=False, draw=True):
    """
    对功能连接(FC)网络进行排序并可视化。

    Args:
        fc_matrix (numpy.ndarray): 功能连接矩阵，形状为 (n, n) 或 (m, n, n)
        electrodes (list): 电极标签列表
        feature_name (str, optional): 特征名称，用于DataFrame列名。默认为'feature'
        ascending (bool, optional): 排序顺序，True为升序，False为降序。默认为False
        draw (bool, optional): 是否绘制热图。默认为True

    Returns:
        tuple: 包含以下元素:
            - mean_values (numpy.ndarray): 每个节点的平均连接强度
            - df_original (pandas.DataFrame): 包含电极和平均值的原始DataFrame
            - df_ranked (pandas.DataFrame): 按平均值排序后的DataFrame
            - rank_indices (numpy.ndarray): 排序后的索引，可用于重新排列原始矩阵
    """
    # 判断输入维度，处理3D和2D矩阵
    if fc_matrix.ndim == 3:
        # 3D矩阵，计算每个节点的平均值
        mean_values = np.mean(np.mean(fc_matrix, axis=0), axis=0)
    elif fc_matrix.ndim == 2:
        # 2D矩阵，直接计算每个节点的平均值
        mean_values = np.mean(fc_matrix, axis=0)
    else:
        raise ValueError(f"输入矩阵维度应为2或3，但得到的是{fc_matrix.ndim}")

    # 确保电极列表长度与矩阵节点数匹配
    if len(electrodes) != len(mean_values):
        raise ValueError(f"电极数量({len(electrodes)})与连接矩阵节点数({len(mean_values)})不匹配")

    # 创建包含电极和平均值的DataFrame
    column_name = f"mean"
    df_original = pd.DataFrame({'electrodes': electrodes, column_name: mean_values})

    # 按均值排序
    df_ranked = df_original.sort_values(column_name, ascending=ascending)

    # 获取排序后的索引，可用于重排原始矩阵
    rank_indices = np.array([list(electrodes).index(e) for e in df_ranked['electrodes']])

    # 绘制热图
    if draw:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # 绘制原始热图
            plt.figure(figsize=(10, 2))
            sns.heatmap([mean_values], cmap='viridis',
                        xticklabels=electrodes, yticklabels=False)
            plt.title(f"original {feature_name} average connection strength")
            plt.tight_layout()
            plt.show()

            # 绘制排序后的热图
            plt.figure(figsize=(10, 2))
            sns.heatmap([df_ranked[column_name]], cmap='viridis',
                        xticklabels=df_ranked['electrodes'], yticklabels=False)
            plt.title(f"sorted {feature_name} average connection strength")
            plt.tight_layout()
            plt.show()

        except ImportError:
            print("Error in drawing heatmap：uninstalled of matplotlib or seaborn")

        except Exception as e:
            print(f"Error in drawing heatmap：{str(e)}")

    return mean_values, df_original, df_ranked, rank_indices

def compute_volume_conduction_factors(distance_matrix, method='exponential', params=None):
    """
    基于距离矩阵计算体积电导效应的因子矩阵。
    支持多种模型：exponential, gaussian, inverse, cutoff, powerlaw, rational_quadratic, generalized_gaussian, sigmoid

    Args:
        distance_matrix (numpy.ndarray): 电极间的距离矩阵，形状为 (n, n)
        method (str): 建模方法
        params (dict): 模型参数字典

    Returns:
        numpy.ndarray: 因子矩阵，与 distance_matrix 同形状
    """
    import numpy as np

    distance_matrix = np.asarray(distance_matrix)
    n = distance_matrix.shape[0]

    # 默认参数集合
    default_params = {
        'exponential': {'sigma': 10.0},
        'gaussian': {'sigma': 5.0},
        'inverse': {'sigma': 5.0, 'alpha': 2.0},
        'cutoff': {'threshold': 5.0, 'factor': 0.5},
        'powerlaw': {'alpha': 2.0},
        'rational_quadratic': {'sigma': 5.0, 'alpha': 1.0},
        'generalized_gaussian': {'sigma': 5.0, 'beta': 2.0},
        'sigmoid': {'mu': 5.0, 'beta': 1.0},
    }

    if params is None:
        if method in default_params:
            params = default_params[method]
        else:
            raise ValueError(f"未提供参数，且方法 '{method}' 没有默认参数")
    elif method in default_params:
        method_params = default_params[method].copy()
        method_params.update(params)
        params = method_params
    else:
        raise ValueError(f"不支持的建模方法: {method}")

    # 初始化结果矩阵
    factor_matrix = np.zeros_like(distance_matrix)
    epsilon = 1e-6  # 防止除0或log0

    if method == 'exponential':
        sigma = params['sigma']
        factor_matrix = np.exp(-distance_matrix / sigma)

    elif method == 'gaussian':
        sigma = params['sigma']
        factor_matrix = np.exp(-np.square(distance_matrix) / (sigma ** 2))

    elif method == 'inverse':
        sigma = params['sigma']
        alpha = params['alpha']
        factor_matrix = 1.0 / (1.0 + np.power(distance_matrix / sigma, alpha))

    elif method == 'cutoff':
        threshold = params['threshold']
        factor = params['factor']
        factor_matrix = np.where(distance_matrix < threshold, factor, 0.0)

    elif method == 'powerlaw':
        alpha = params['alpha']
        factor_matrix = 1.0 / (np.power(distance_matrix, alpha) + epsilon)

    elif method == 'rational_quadratic':
        sigma = params['sigma']
        alpha = params['alpha']
        factor_matrix = np.power(1.0 + (np.square(distance_matrix) / (2 * alpha * sigma ** 2)), -alpha)

    elif method == 'generalized_gaussian':
        sigma = params['sigma']
        beta = params['beta']
        factor_matrix = np.exp(-np.power(distance_matrix / sigma, beta))

    elif method == 'sigmoid':
        mu = params['mu']
        beta = params['beta']
        factor_matrix = 1.0 / (1.0 + np.exp((distance_matrix - mu) / beta))

    else:
        raise ValueError(f"不支持的体积电导建模方法: {method}")

    # 对角线置为1（自我连接）
    np.fill_diagonal(factor_matrix, 1.0)
    return factor_matrix

def compute_volume_conduction_factors_(distance_matrix, method='exponential', params=None):
    """
    基于距离矩阵计算体积电导效应的因子矩阵。
    
    Args:
        distance_matrix (numpy.ndarray): 电极间的距离矩阵，形状为 (n, n)
        method (str, optional): 体积电导效应建模方法。默认为'exponential'。
            - 'exponential': f(d) = exp(-d/sigma)，距离越近，影响越大
            - 'gaussian': f(d) = exp(-d²/sigma²)，高斯衰减模型
            - 'inverse': f(d) = 1/(1 + (d/sigma)^alpha)，反比例衰减模型
            - 'cutoff': 如果距离小于阈值，则使用固定因子，否则为0
        params (dict, optional): 建模参数，默认为None，使用默认参数。
            - 'exponential'需要参数'sigma'，控制衰减率
            - 'gaussian'需要参数'sigma'，控制高斯分布宽度
            - 'inverse'需要参数'sigma'和'alpha'，控制衰减率和衰减形状
            - 'cutoff'需要参数'threshold'和'factor'，分别指定截断距离和固定因子值
    
    Returns:
        numpy.ndarray: 体积电导效应因子矩阵，形状与输入距离矩阵相同，对角线元素为1
    """
    import numpy as np
    
    # 确保输入是numpy数组
    distance_matrix = np.asarray(distance_matrix)
    n = distance_matrix.shape[0]
    
    # 设置默认参数
    default_params = {
        'exponential': {'sigma': 10.0},
        'gaussian': {'sigma': 5.0},
        'inverse': {'sigma': 5.0, 'alpha': 2.0},
        'cutoff': {'threshold': 5.0, 'factor': 0.5}
    }
    
    # 如果提供了参数，更新默认参数
    if params is None:
        if method in default_params:
            params = default_params[method]
        else:
            raise ValueError(f"未提供参数，且方法 '{method}' 没有默认参数")
    elif method in default_params:
        # 只更新提供的参数，保留其他默认参数
        method_params = default_params[method].copy()
        method_params.update(params)
        params = method_params
            
    # 创建因子矩阵
    factor_matrix = np.zeros_like(distance_matrix)
    
    # 根据选择的方法计算因子
    if method == 'exponential':
        sigma = params['sigma']
        # 指数衰减模型: exp(-d/sigma)
        factor_matrix = np.exp(-distance_matrix / sigma)
        
    elif method == 'gaussian':
        sigma = params['sigma']
        # 高斯衰减模型: exp(-d²/sigma²)
        factor_matrix = np.exp(-(distance_matrix**2) / (sigma**2))
        
    elif method == 'inverse':
        sigma = params['sigma']
        alpha = params['alpha']
        # 反比例衰减模型: 1/(1 + (d/sigma)^alpha)
        factor_matrix = 1.0 / (1.0 + (distance_matrix / sigma)**alpha)
        
    elif method == 'cutoff':
        threshold = params['threshold']
        factor = params['factor']
        # 截断模型: 如果距离小于阈值，则为固定因子，否则为0
        factor_matrix = np.where(distance_matrix < threshold, factor, 0.0)
        
    else:
        raise ValueError(f"不支持的体积电导建模方法: {method}")
    
    # 设置对角线元素为1（自身对自身的影响为1）
    np.fill_diagonal(factor_matrix, 1.0)
    
    return factor_matrix

import matplotlib.pyplot as plt
def draw_weight_mapping(ranking, ranked_values, ranked_electrodes=None, offset=0, transformation=None, reverse=False):
    weight_mean = ranked_values
    if reverse:
        weight_mean = 1 - weight_mean
    distribution = utils_feature_loading.read_distribution('seed')
    
    dis_t = distribution.iloc[ranking]
    
    x = np.array(dis_t['x'])
    y = np.array(dis_t['y'])
    electrodes = dis_t['channel']
    
    # 归一化 label_driven_mi_mean 以适应颜色显示（假设它是数值列表）
    if transformation == 'log':
        values = np.array(np.log(weight_mean) + offset)
    else: 
        values = np.array(weight_mean + offset)
    
    # 绘制散点图
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(x, y, c=values, cmap='coolwarm', s=100, edgecolors='k')
    
    # 添加颜色条
    cbar = plt.colorbar(sc)
    cbar.set_label('Label Driven MI Mean')
    
    # 标注电极通道名称
    for i, txt in enumerate(electrodes):
        plt.text(x[i], y[i], txt, fontsize=9, ha='right', va='bottom')
    
    # 设置标题和坐标轴
    plt.title("Label Driven MI Mean Distribution on Electrodes")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plt.show()
    
    return weight_mean

if __name__ == '__main__':
    # %% Load Distance Matrix
    # dm_pcc_alpha, dm_pcc_beta, dm_pcc_gamma, dm_pcc_joint = load_global_averages(feature='PCC')
    # dm_plv_alpha, dm_plv_beta, dm_plv_gamma, dm_plv_joint = load_global_averages(feature='PLV')

    # %% Distance Matrix of Euclidean
    # channel_names, distance_matrix = compute_distance_matrix('seed')
    # utils_visualization.draw_projection(distance_matrix)

    # channel_names, distance_matrix_ste = compute_distance_matrix('seed', method='stereo')
    # utils_visualization.draw_projection(distance_matrix_ste)

    # %% Factor Matrix
    # factor_matrix = compute_volume_conduction_factors(distance_matrix)
    # utils_visualization.draw_projection(factor_matrix)
    
    # factor_matrix_ste = compute_volume_conduction_factors(distance_matrix_ste)
    # utils_visualization.draw_projection(factor_matrix_ste)

    # %% Distance Matrix of PCC
    # # global_alpha_average, global_beta_average, global_gamma_average, global_joint_average = compute_averaged_fcnetwork('PCC', subjects=range(1, 16), draw=True, save=True)
    # global_alpha_average, global_beta_average, global_gamma_average, global_joint_average = load_global_averages(
    #     feature='PCC')

    # joint = global_joint_average
    # joint_mean_1d = np.mean(joint, axis=0)

    # # get electrodes
    # distribution = utils_feature_loading.read_distribution('seed')
    # electrodes = distribution['channel']

    # mean_values, df_original, df_ranked, rank_indices = rank_and_visualize_fc_network(global_joint_average, electrodes,
    #                                                                                   feature_name='PCC')
    
    # %% Matrix of differ(Connectivity_Matrix_PCC, Factor_Matrix); stereo distance matrix; generalized_gaussian
    # Target
    import channel_selection_weight_mapping
    channel_selection_weight_mapping.draw_weight_mapping(ranking_method='label_driven_mi')
    
    # Fitted
    channel_names, distance_matrix = compute_distance_matrix('seed', method='stereo')
    distance_matrix = normalize_matrix(distance_matrix)
    # utils_visualization.draw_projection(distance_matrix)

    factor_matrix = compute_volume_conduction_factors(distance_matrix, method='generalized_gaussian', params={'sigma': 2.27, 'beta': 5.0})
    factor_matrix = normalize_matrix(factor_matrix)
    # utils_visualization.draw_projection(factor_matrix)

    _, _, _, global_joint_average = load_global_averages(feature='PCC')
    global_joint_average = normalize_matrix(global_joint_average)
    # utils_visualization.draw_projection(global_joint_average)

    differ_PCC_DM = global_joint_average - factor_matrix
    # utils_visualization.draw_projection(differ_PCC_DM)

    # get electrodes
    distribution = utils_feature_loading.read_distribution('seed')
    electrodes = distribution['channel']

    # **********************REVISE HERE
    _, _, df_ranked, rank_indices = rank_and_visualize_fc_network(differ_PCC_DM, electrodes, feature_name='PCC') #, exclude_electrodes=['CB1', 'CB2'])
    # **********************************

    draw_weight_mapping(rank_indices, df_ranked['mean'])