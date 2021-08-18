import cv2
import os
from configs import config_preprocessing as config
from base.utils import load_single_pkl, txt_row_count
import pandas as pd
import numpy as np


def get_video_path(label, partition):
    label = label.split(".txt")[0]
    video_filename_without_extension = label

    # Find the corresonding video filename.
    if video_filename_without_extension.endswith("_right"):
        video_filename_without_extension = video_filename_without_extension.split("_right")[0]
    elif video_filename_without_extension.endswith("_left"):
        video_filename_without_extension = video_filename_without_extension.split("_left")[0]

    if (video_filename_without_extension + ".avi") in video_to_partition_dict[partition]:
        corresponding_video_filename = video_filename_without_extension + ".avi"
    elif (video_filename_without_extension + ".mp4") in video_to_partition_dict[partition]:
        corresponding_video_filename = video_filename_without_extension + ".mp4"
    else:
        print(video_filename_without_extension)
        raise ValueError("Cannot find the corresponding video")

    corresponding_video = os.path.join(raw_video_path, corresponding_video_filename)

    return corresponding_video

raw_video_path = config['raw_video_path']
annotation_path = config['annotation_path']

dataset_info =load_single_pkl(r"H:\Affwild2_processed", "dataset_info")
video_to_partition_dict = dataset_info['video_to_partition_dict']
annotation_to_partition_dict = dataset_info['annotation_to_partition_dict']

available_frame_indices = load_single_pkl(r"H:\Affwild2_processed", "available_frame_indices")
labeled_frame_indices = load_single_pkl(r"H:\Affwild2_processed", "labeled_frame_indices")

count = 1
result = []
for partition, trials in annotation_to_partition_dict.items():
    if partition == "Target_Set":
        for trial in trials:
            trial_name = trial
            current_annotation_path = os.path.join(annotation_path, partition, trial)
            current_video_path = get_video_path(trial, partition)
            video = cv2.VideoCapture(current_video_path)
            num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            num_considered = np.count_nonzero(np.asarray(available_frame_indices[partition][trial]) == 1)
            difference = num_frames - num_considered

            result.append([trial_name, num_frames, num_considered, difference])
    else:
        for trial in trials:
            trial_name = trial[:-4]
            current_annotation_path = os.path.join(annotation_path, partition, trial)
            current_video_path = get_video_path(trial, partition)
            video = cv2.VideoCapture(current_video_path)
            num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            num_considered = np.count_nonzero(np.asarray(available_frame_indices[partition][trial_name]) == 1)
            difference = num_frames - num_considered

            result.append([trial_name, num_frames, num_considered, difference])
                # string = "{}: num_frames = {:d}, num_rows = {:d}, num_inequal = {:d}".format(trial_name, num_frames, num_labels, count)
                # print(string)

result = np.stack(result)
df = pd.DataFrame(result, columns=["trial name", "frame_count", "considered_count", "difference"])
df.to_csv("length.csv")




print(0)