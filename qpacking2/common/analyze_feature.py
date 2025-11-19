"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/5/1

# Description: Feature distribution analyze and visualization.
# ------------------------------------------------------------------------------
"""
import os
import pickle
from tqdm import tqdm
import numpy as np

from collections import Counter
from math import comb

from qpacking2.common import logger

logger = logger.setup_log(name=__name__)

def load_existing_results(pkl_file):
    """
    Load existing results from a pickle file.
    :param pkl_file: existing results file
    :return: a dictionary containing loaded results
    """
    try:
        with open(pkl_file, "rb") as f:
            results_dict = pickle.load(f)  # output file only 1 obj.
            if not isinstance(results_dict, dict):
                return {}
            return results_dict
    except (FileNotFoundError, EOFError):
        logger.error('FileNotFoundError')
        return {}
    except Exception as e:
        logger.error(e)
        return {}
def load_pkl(pkl_file):
    with open(pkl_file, 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data

def analyze_class(load_existing_results):
    binary_positive_stats = []
    contrastive_same_cluster_stats = []
    for protein_id, feature in tqdm(load_existing_results.items()):
        # binary stats
        class_feature = feature['class']
        seq_length = len(feature['area'])  # 'area' contains features of all residues
        hydrophobic_residue_num = len(class_feature)
        positive_hydrophobic_ratio = hydrophobic_residue_num / seq_length
        binary_positive_stats.append(positive_hydrophobic_ratio)

        # contrastive stats
        counts = Counter(class_feature.values())
        total_pairs = comb(len(class_feature), 2)  # n(p,k) >= 3
        same_pairs = sum(comb(size, 2) for size in counts.values())
        same_cluster_ratio = same_pairs / total_pairs
        contrastive_same_cluster_stats.append(same_cluster_ratio)

    mean_binary_positive_stats = np.mean(binary_positive_stats)
    mean_same_cluster_ratio = np.mean(contrastive_same_cluster_stats)
    print('mean_binary_positive_stats: ', mean_binary_positive_stats)
    print('mean_binary_negative_stats: ', str(1 - mean_binary_positive_stats))
    print()
    print('mean_same_cluster_ratio: ', mean_same_cluster_ratio)
    print('mean_different_cluster_ratio: ', str(1 - mean_same_cluster_ratio))

def prepare_plot_feature(load_existing_results, feature_name='centrality', bins=10):
    data = [i for protein_id, feature in load_existing_results.items() for i in feature[feature_name].values()]
    counts_density, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_widths = np.diff(bin_edges)
    print(f"{feature_name} feature")

    bin_centers = bin_edges[:-1] + bin_widths / 2
    print("bin_centers:")
    for i in bin_centers:
        print(i)
    print()

    frequencies = counts_density * bin_widths
    print("frequencies:")
    for i in frequencies:
        print(i)

    # print("Sum of frequencies:", frequencies.sum())
