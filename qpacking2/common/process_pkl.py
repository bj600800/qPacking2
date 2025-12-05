"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/12/5

# Description: Multitask training esm2
# ------------------------------------------------------------------------------
"""
import os
import math
import numpy as np
from tqdm import tqdm
import pickle
from qpacking2.common import logger
from qpacking2.common.statis_plot_feature import plot_feature

logger = logger.setup_log(name=__name__)

residue_3to1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
    "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
}

def load_pkl(pkl_file):
    with open(pkl_file, 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data

def load_dict_pkl(pkl_file):
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

def construct_feature(feature):
    output_feature = []
    for protein_name, feature_dict in tqdm(feature.items()):
        seq_length = feature_dict['length']
        res_name_dict = feature_dict['resid_name']
        sequence = ''.join([residue_3to1[res_name_dict[k]] for k in sorted(res_name_dict.keys())])

        first_res_id = min(res_name_dict.keys())
        for feature_name, feature in feature_dict.items():
            if feature_name in ['class', 'rsa', 'bsa', 'degree', 'order']:
                data = {}
                data['protein_name'] = protein_name
                data['length'] = seq_length
                data['sequence'] = sequence
                data['feature_name'] = feature_name

                if feature_name == 'class':
                    position_label = [0] * seq_length
                    data['feature_name'] = 'position'
                    for pos, cluster_id in feature.items():
                        idx = pos - first_res_id
                        position_label[idx] = 1
                    data['label'] = position_label

                elif feature_name == 'rsa':
                    rsa_label = [0] * seq_length
                    for pos, rsa_value in feature.items():
                        idx = pos - first_res_id
                        rsa_label[idx] = 1 if rsa_value < 0.05 else 0
                    data['label'] = rsa_label

                elif feature_name in ['bsa', 'order']:
                    label = [-100] * seq_length
                    for pos, value in feature.items():
                        idx = pos - first_res_id
                        label[idx] = value / seq_length
                    data['label'] = label

                else:
                    # degree
                    label = [-100] * seq_length
                    for pos, value in feature.items():
                        idx = pos - first_res_id
                        label[idx] = value
                    data['label'] = label

                output_feature.append(data)

    return output_feature


def get_example_data(input_pkl, output_pkl):
    existing_results = load_dict_pkl(input_pkl)
    output = {}
    for protein_name, feature in tqdm(existing_results.items()):
        output[protein_name] = feature
        if len(output) == 10:
            break


if __name__ == '__main__':
    # feature_pkl = r"/Users/douzhixin/Developer/qPacking-esm/data/feature/feature.pkl"
    # output_pkl = r"/Users/douzhixin/Developer/qPacking2/data/feature/structure_feature.pkl"
    # ret = load_dict_pkl(feature_pkl)
    # output_feature = construct_feature(ret)
    # with open(output_pkl, "wb") as f:
    #     pickle.dump(output_feature, f)

    feature_pkl = r"/Users/douzhixin/Developer/qPacking2/data/feature/structure_feature.pkl"
    ret = load_pkl(feature_pkl)
    print(ret)
    print(len(ret))






