import os
import numpy as np
from tqdm import tqdm

from base.utils import load_single_pkl, save_pkl_file
from base.dataset import ABAW2_VA_Arranger
from experiment_regular import Experiment

# Please change it accordingly.
dataset_path = "/home/zhangsu/dataset/affwild2"


# No need to change the rest.
feature_list = ['vggish', 'mfcc', 'egemaps']
dataset_info = load_single_pkl(directory=dataset_path, filename="dataset_info")
partition_list = ['Train_Set', 'Validation_Set', 'Target_Set']
mean_std_dict = {feature: {fold: {partition: {'mean': None, 'std': None} for partition in partition_list} for fold in range(6)} for feature in feature_list}
feature_array = []
Experiment.init_random_seed()

for feature in feature_list:
    for fold in range(6):
        partition_dict = ABAW2_VA_Arranger.generate_partition_dict_for_cross_validation(dataset_info['partition'], fold)
        for partition, trials in partition_dict.items():
            lengths = 0
            sums = 0
            for trial in tqdm(trials, total=len(trials)):
                feature_path = os.path.join(dataset_path, "npy_data", trial, feature + ".npy")
                feature_matrix = np.load(feature_path)
                feature_array = feature_matrix.flatten()
                lengths += len(feature_array)
                sums += feature_array.sum()
            mean_std_dict[feature][fold][partition]['mean'] = sums / (lengths + 1e-10)


for feature in feature_list:
    for fold in range(6):
        partition_dict = ABAW2_VA_Arranger.generate_partition_dict_for_cross_validation(dataset_info['partition'], fold)
        for partition, trials in partition_dict.items():
            lengths = 0
            mean = mean_std_dict[feature][fold][partition]['mean']
            x_minus_mean_square = 0
            for trial in tqdm(trials, total=len(trials)):
                feature_path = os.path.join(dataset_path, "npy_data", trial, feature + ".npy")
                feature_matrix = np.load(feature_path)
                feature_array = feature_matrix.flatten()
                lengths += len(feature_array)
                x_minus_mean_square += np.sum((feature_array - mean) ** 2)
            x_minus_mean_square_divide_N_minus_1 = x_minus_mean_square / (lengths - 1)
            mean_std_dict[feature][fold][partition]['std'] = np.sqrt(x_minus_mean_square_divide_N_minus_1)

save_pkl_file(dataset_path, "mean_std_dict.pkl", mean_std_dict)