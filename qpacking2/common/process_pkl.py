"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/11/18

# Description: protein feature matrix
# ------------------------------------------------------------------------------
"""
import os
from tqdm import tqdm
import pickle
from qpacking2.common.analyze_feature import load_existing_results

residue_3to1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
    "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
}

def get_pkl(pkl1, pkl2):
    """
    dict_keys(['class', 'rsa', 'bsa', 'degree', 'order', 'resid_name', 'length'])
    Args:
        pkl1:
        pkl2:

    Returns:

    """
    feature1 = load_existing_results(pkl1)
    feature2 = load_existing_results(pkl2)

    output_pkl = {**feature1, **feature2}
    return output_pkl

def split_feature(feature, key, data_type):
    single_feature = {}
    for k, v in feature.items():
        if key=='class':
            single_feature[k] = v[key]
        elif key=='order':
            length = v['length']
            single_feature[k] = {key: value/length for key, value in v[key].items()}
        elif key=='resid_name':
            seq_dict = {key: residue_3to1[value] for key, value in v[key].items()}
            sequence = ''.join([seq_dict[k] for k in sorted(seq_dict.keys())])
            single_feature[k] = {'seq': sequence, 'seq_dict': seq_dict}
        else:
            if data_type == 'float32':
                single_feature[k] = {key: float(value) for key, value in v[key].items()}
            elif data_type == 'int':
                single_feature[k] = {key: value for key, value in v[key].items()}
    return single_feature

def run_split(input_pkl):
    dir_path = os.path.dirname(input_pkl)
    file_name = os.path.basename(input_pkl)
    existing_results = load_existing_results(input_pkl)
    feature_names = {'class': 'int', 'bsa': 'float32', 'degree': 'int', 'rsa': 'float32', 'order': 'float32', 'resid_name': 'str'}
    for name, data_type in tqdm(feature_names.items()):
        new_pkl = os.path.join(dir_path, file_name.split('.')[0]+f'_{name}.pkl')
        new_feature = split_feature(existing_results, name, data_type)
        with open(new_pkl, "wb") as f:
            pickle.dump(new_feature, f)

def get_example_data(input_pkl):
    dir_path = os.path.dirname(input_pkl)
    file_name = os.path.basename(input_pkl)
    existing_results = load_existing_results(input_pkl)
    feature_names = {'class': 'int', 'bsa': 'float32', 'degree': 'int', 'rsa': 'float32', 'order': 'float32',
                     'resid_name': 'str'}
    test_protein_name = []
    for name, data_type in tqdm(feature_names.items()):
        new_pkl = os.path.join(dir_path, f'test/{file_name.split(".")[0]}_{name}.pkl')
        os.makedirs(os.path.dirname(new_pkl), exist_ok=True)
        new_feature = split_feature(existing_results, name, data_type)
        if name == 'class':
            test_protein_name = list(new_feature.keys())[:10]
        save_feature = {protein_name: feature for protein_name, feature in new_feature.items() if
                        protein_name in test_protein_name}
        with open(new_pkl, "wb") as f:
            pickle.dump(save_feature, f)

def run(pkl1, pkl2, output_pkl_path):
    output_pkl = get_pkl(pkl1, pkl2)
    with open(output_pkl_path, "wb") as f:
        pickle.dump(output_pkl, f)
    run_split(output_pkl_path)


if __name__ == '__main__':
    # pkl1 = r"/Users/douzhixin/Developer/qPacking/Data/feature/70_feature/70_results.pkl"
    # pkl2 = r"/Users/douzhixin/Developer/qPacking-esm/data/feature/80/80_hydrophobic_feature.pkl"
    # output_pkl_path = r"/Users/douzhixin/Developer/qPacking-esm/data/feature/all/feature.pkl"
    # run(pkl1, pkl2, output_pkl_path)
    # print(load_existing_results(test_pkl))
    # get_example_data(output_pkl_path)
    test_pkl_dir = r"/Users/douzhixin/Developer/qPacking-esm/data/feature/all/test"
    test_pkl_file = [os.path.join(test_pkl_dir, file) for file in os.listdir(test_pkl_dir)]
    for file in test_pkl_file:
        print(file)
        print(load_existing_results(file))

