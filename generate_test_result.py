import cv2
import os
from configs import config_preprocessing as config
from base.utils import load_single_pkl, txt_row_count
import pandas as pd
import numpy as np

dataset_path = r"/home/zhangsu/dataset/affwild2"
output_load_path = r"/home/zhangsu/ABAW2-attention/save"
txt_save_path = r"/home/zhangsu/ABAW2-attention/save/unimodal_fold0"
os.makedirs(txt_save_path, exist_ok=True)
output_folders = "cv"
emotions = ["Valence", "Arousal"]
folds = [0]

dataset_info =load_single_pkl(dataset_path, "dataset_info")
video_to_partition_dict = dataset_info['video_to_partition_dict']
annotation_to_partition_dict = dataset_info['annotation_to_partition_dict']

available_frame_indices = load_single_pkl(dataset_path, "available_frame_indices")
labeled_frame_indices = load_single_pkl(dataset_path, "labeled_frame_indices")
test_data_list = sorted(pd.read_csv(os.path.join(dataset_path, "test_set_VA_Challenge.txt"), header=None).values[:,0])
count = 1
result = []
for partition, trials in annotation_to_partition_dict.items():
    if partition == "Target_Set":
        for test_trial, guess_trial in zip(test_data_list, trials):
            assert test_trial == guess_trial

#         # for trial in trials:

            trial_name = test_trial

            available_index = np.asarray(available_frame_indices[partition][trial_name])
            to_pad = np.squeeze(np.where(available_index == 0)[0])
            to_assign = np.squeeze(np.where(available_index == 1)[0])

            result_matrix = np.empty((len(folds), len(available_index), 2))
            for i, fold in enumerate(folds):
                for j, emotion in enumerate(emotions):
                    output_path = os.path.join(output_load_path, output_folders, str(fold), emotion, trial_name+".txt")
                    pred = pd.read_csv(output_path, header=None, skiprows=1).values[:, 0]
                    assert len(available_index) == len(pred)

                    pred = np.clip(pred, -1, 1)
                    result_matrix[i, :, j] = pred

                if len(result_matrix) > 1:
                    pass

                df = pd.DataFrame(data=result_matrix[0, :, :], index=None, columns={"valence", "arousal"})
                txt_path = os.path.join(txt_save_path, trial_name + ".txt")
                df.to_csv(txt_path, sep=",", index=None)
