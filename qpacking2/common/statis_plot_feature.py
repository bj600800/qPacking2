"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/12/5

# Description: Multitask training esm2
# ------------------------------------------------------------------------------
"""
from scipy.stats import skew, kurtosis
import os
import pickle
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from math import comb
from qpacking2.common import logger

logger = logger.setup_log(name=__name__)

def analyze_class(load_existing_results):
    total = 0
    hydro = 0
    for protein_id, feature in tqdm(load_existing_results.items()):
        # binary stats
        seq_length = feature['length']
        class_feature = feature['class']
        hydrophobic_residue_num = len(class_feature)
        total += seq_length
        hydro += hydrophobic_residue_num

    positive = hydro / total
    negative = 1 - positive

    print(f"Hydrophobic: {positive}")
    print(f"Non-hydrophobic: {negative}")

def analyze_rsa(load_existing_results):
    total = 0
    interior = 0

    for protein_id, feature in tqdm(load_existing_results.items()):
        # binary stats
        seq_length = feature['length']
        rsa_feature = feature['rsa']
        interior_len = len([i for i in rsa_feature.values() if i < 0.05])
        total += seq_length
        interior += interior_len

    interior = interior / total
    surface = 1 - interior
    print(f"Interior: {interior}")
    print(f"Surface: {surface}")


def zscore_list(data):
    mean = sum(data) / len(data)
    std = (sum((x - mean)**2 for x in data) / len(data))**0.5
    datalist = [(x - mean) / std for x in data]
    return datalist

def plot_feature(load_existing_results, feature_name, dtype=None, bins=11):
    if dtype == 'single':
        data = [i for protein_id, feature in load_existing_results.items()
                for i in feature.values()]
    else:
        data = [i for protein_id, feature in load_existing_results.items()
                for i in feature[feature_name].values()]

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

    # 绘图
    plt.figure(figsize=(6, 4))
    plt.bar(bin_centers, frequencies, width=bin_widths,
            edgecolor="black", color="#FFB050",
            alpha=0.7, linewidth=1)

    plt.xlabel(feature_name)
    plt.ylabel("Frequency")
    plt.title(f"Frequency distribution of {feature_name}")
    plt.tight_layout()
    figure_path = rf"/Users/douzhixin/Developer/qPacking-esm/figure/python/feature/{feature_name}.png"
    plt.savefig(figure_path, dpi=600, bbox_inches='tight')
    plt.show()

