import cv2
import os
from configs import config_preprocessing as config
from base.utils import load_single_pkl, txt_row_count
import pandas as pd
import numpy as np
from base.logger import ConcordanceCorrelationCoefficient
import matplotlib.pyplot as plt

dataset_path = r"/home/zhangsu/dataset/affwild2"
output_load_path = r"/home/zhangsu/ABAW2-attention/save"
txt_save_path = r"/home/zhangsu/ABAW2-attention/save/unimodal_fold0_late_clip"
os.makedirs(txt_save_path, exist_ok=True)
output_folders = "cv_v3"
emotions = ["Valence", "Arousal"]
folds = [0,1,2,3,4,5]

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

            trial_name = test_trial

            available_index = np.asarray(available_frame_indices[partition][trial_name])

            result_matrix = np.empty((len(folds), len(available_index), 2))
            for i, fold in enumerate(folds):
                for j, emotion in enumerate(emotions):
                    output_path = os.path.join(output_load_path, output_folders, str(fold), emotion, trial_name+".txt")
                    pred = pd.read_csv(output_path, header=None, skiprows=1).values[:, 0]
                    assert len(available_index) == len(pred)

                    # Early clip
                    # pred = np.clip(pred, -1, 1)

                    result_matrix[i, :, j] = pred

            if len(result_matrix) == 1:
                final_result = result_matrix[0, :, :]

            else:
                final_result = []
                for p, emotion in enumerate(emotions):
                    # plt.figure()
                    # for data in result_matrix[:, :, p]:
                    #     plt.plot(data)
                    #
                    # plt.show()

                    ccc_centering_handler = ConcordanceCorrelationCoefficient(result_matrix[:, :, p])

                    centered_data = np.mean(ccc_centering_handler.centered_data, axis=0)
                    mean_data = np.mean(result_matrix[:, :, p], axis=0)

                    # plt.figure()
                    #
                    # plt.plot(mean_data)
                    # plt.show()
                    #
                    # print(0)
                    # plt.figure()
                    #
                    # plt.plot(centered_data)
                    # plt.show()
                    #
                    # plt.figure()
                    #
                    # plt.plot(np.clip(centered_data, -1, 1))
                    # plt.show()


                    final_result.append(centered_data)
                final_result = np.stack(final_result).T

            # Late clip
            final_result = np.clip(final_result, -1, 1)

            df = pd.DataFrame(data=final_result, index=None, columns={"valence", "arousal"})
            txt_path = os.path.join(txt_save_path, trial_name + ".txt")
            df.to_csv(txt_path, sep=",", index=None)

            print(count)
            count += 1
