"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/12/5

# Description: Multitask training esm2
# ------------------------------------------------------------------------------
"""
from tqdm import tqdm
import numpy as np
from qpacking2.common.process_pkl import load_pkl
from qpacking2.common import logger

logger = logger.setup_log(name=__name__)

def format_position_binary(feature_pkl):
    result = {
        "200-250": [],
        "250-300": [],
        "300-350": [],
        "350-400": [],
    }

    data = load_pkl(feature_pkl)

    for feature in tqdm(data, desc="Processing features"):
        feature_name = feature['feature_name']
        ret = {}
        if feature_name == 'position':
            protein_name = feature['protein_name']
            length = feature['length']
            label = feature['label']
            pos_count = label.count(1)
            neg_count = label.count(0)
            ret['protein_name'] = protein_name
            ret['length'] = length
            ret['pos_count'] = pos_count
            ret['neg_count'] = neg_count
            ret['pos_ratio'] = pos_count / length

            if 200 <= length < 250:
                result["200-250"].append(ret)
            elif 250 <= length < 300:
                result["250-300"].append(ret)
            elif 300 <= length < 350:
                result["300-350"].append(ret)
            elif 350 <= length <= 400:
                result["350-400"].append(ret)
    return result

def sample_seq_level_soft(
    result,
    ratio_dict=None,
    extra_ratio=0.05,
    seed=3407
):
    """
    two stage downsample：
    1) keep bin front ratio_dict[bin] high pos-ratio sequence
    2) fill with remaining sequences
    """

    if ratio_dict is None:
        ratio_dict = {
            '200-250': 0.3,
            '250-300': 0.3,
            '300-350': 0.1,
            '350-400': 0.1
        }

    np.random.seed(seed)

    keep_protein = {}
    bin_result = {}
    keep_protein_names = []
    for bin_name, proteins in result.items():

        # sort
        proteins_sorted = sorted(
            proteins,
            key=lambda x: x['pos_ratio'],
            reverse=True
        )

        # keep the front % data
        n_keep = int(len(proteins_sorted) * ratio_dict[bin_name])
        stage1 = proteins_sorted[:n_keep]

        # fill with the remaining dataset
        remain = proteins_sorted[n_keep:]
        n_extra = int(len(remain) * extra_ratio)
        if n_extra > 0:
            stage2 = list(np.random.choice(remain, n_extra, replace=False))
        else:
            stage2 = []

        final = stage1 + stage2
        keep_protein[bin_name] = final

        keep_protein_names.extend([p['protein_name'] for p in final])

        bin_pos = sum(p['pos_count'] for p in final)
        bin_neg = sum(p['neg_count'] for p in final)
        total = bin_pos + bin_neg
        bin_ratio = bin_pos / total if total > 0 else None

        bin_result[bin_name] = {
            'count': {'pos_count': bin_pos, 'neg_count': bin_neg},
            'pos_ratio': bin_ratio
        }


    all_pos = sum(b['count']['pos_count'] for b in bin_result.values())
    all_neg = sum(b['count']['neg_count'] for b in bin_result.values())
    total_ratio = all_pos / (all_pos + all_neg)

    print("complete downsample")
    print('protein count:', len(keep_protein_names))
    print(bin_result)
    print("total positive:", all_pos)
    print("total negative:", all_neg)
    print("pos_ratio:", total_ratio)
    print("neg/pos:", all_neg / all_pos)



    return keep_protein_names


if __name__ == '__main__':
    pkl_file = r"/Users/douzhixin/Developer/qPacking2/data/feature/structure_feature.pkl"
    # pkl_file = r"/Users/douzhixin/Developer/qPacking2/data/test/feature/example_feature.pkl"
    result = format_position_binary(pkl_file)
    sample_seq_level_soft(result)



