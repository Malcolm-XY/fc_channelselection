# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 18:18:41 2025

@author: usouu
"""
import numpy as np

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

import fc_computer as fc

def hierarchical_clustering(correlation_matrix, threshold=0.7):
    """
    使用层次聚类划分信号
    :param correlation_matrix: 相关系数矩阵
    :param threshold: 不相似的阈值
    :return: clusters: ndarray of cluster labels
    """
    distance_matrix = 1 - correlation_matrix
    np.fill_diagonal(distance_matrix, 0)  # 自己与自己的距离设为 0
    condensed_dist = squareform(distance_matrix, checks=False)  # 转换为压缩形式
    linkage_matrix = linkage(condensed_dist, method='average')  # 层次聚类
    clusters = fcluster(linkage_matrix, threshold, criterion='distance')
    return clusters

from collections import defaultdict

def parse_clusters(cluster_labels):
    """
    将聚类标签解析为信号分组
    :param cluster_labels: 聚类标签列表
    :return: group_dict: 按标签分组的信号索引
    """
    group_dict = defaultdict(list)
    for idx, label in enumerate(cluster_labels):
        group_dict[label].append(idx)
    return group_dict

fc_alpha, fc_beta, fc_gamma = fc.load_global_averages()

# 使用层次聚类划分
cluster_labels = hierarchical_clustering(fc_gamma, threshold=0.5)
print("层次聚类结果:", cluster_labels)

# 解析聚类结果
groups = parse_clusters(cluster_labels)

# 打印分组结果
for label, indices in groups.items():
    print(f"聚类 {label}: 信号索引 {indices}")