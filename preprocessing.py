import os
import pickle
import subprocess

import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
from PIL import Image

from configs import config_preprocessing as config
from base.utils import OpenFaceController, txt_row_count, standardize_facial_landmarks, save_pkl_file


class ABAW2_Preprocessing(object):
    def __init__(self):
        self.raw_video_path = config['raw_video_path']
        self.test_video_path = config['test_video_path']
        self.image_path = config['image_path']
        self.annotation_path = config['annotation_path']
        self.output_path = config['output_path']

        self.aural_feature_list = config['aural_feature_list']
        self.opensmile_exe_path = config['opensmile_exe_path']
        self.opensmile_config_path = config['opensmile_config_path']

        self.openface_config = config['openface_config']

        self.dataset_info = {}

        # Step 1: obtain the data list for training, validation set, and test sets.
        # Step 1.1: obtain the video list.
        self.video_to_partition_dict = self.generate_video_to_partition_dict()
        self.dataset_info['video_to_partition_dict'] = self.video_to_partition_dict

        # Step 1.2: obtain the annotation list.
        self.annotation_to_partition_dict = self.generate_annotation_to_partition_dict()
        self.dataset_info['annotation_to_partition_dict'] = self.annotation_to_partition_dict

        # # Step 2: preprocess the aural modality.
        # # Step 2.1: convert all mp4 or avi to uncompressed wav format, since the OpenSmile is not compiled
        # #   with ffmpeg support. The wav files are to be stored in another folder with the same filename.
        # self.convert_video_to_wav()
        #
        # # # Step 2.2: extract aural features, including mfcc, egemaps, and vggish.
        # self.extract_aural_features()

        # Step 3: preprocess the visual modality.
        # Step 3.1: obtain the labeled frame index for each video.
        # 1 for labeled, 0 for non-labeled.
        self.labeled_frame_indices_of_each_video = self.get_labeled_frame_indices()

        # Step 3.2: obtained the available frame images from "cropped_aligned" folder.
        # 1 for available, 0 for unavailable, -1 for non-labeled.
        self.available_frame_indices_of_each_video = self.get_available_frame_indices()

        # # Step 3.3: save the labels in npy format.
        # self.extract_visual_features()

        # Step 4: prepare the extracted features into npy format.
        # Step 4.1: compact the downloaded cropped_aligned jpgs to npy format.
        #       And filled the missing jpg as black images.
        self.compact_visual_features()
        #
        # # Step 4.2: compact the extracted aural features, including mfcc, egemaps, and vggish.
        # self.compact_aural_features()
        #
        # # Step 4.3: compact the txt labels into npy format.
        # self.compact_labels()
        #
        # Step 5: generate the dataset info.
        self.generate_dataset_info()

    def generate_dataset_info(self):
        dataset_info = {}

        partition_dict = {'Train_Set': {}, 'Validation_Set': {}, 'Test_Set': {}}
        for partition, labels in self.annotation_to_partition_dict.items():
            for label in tqdm(labels, total=len(labels), desc=partition):
                trial_name = label[:-4]
                if partition == "Test_Set":
                    trial_name = label
                trial_length = np.sum(self.labeled_frame_indices_of_each_video[partition][trial_name])
                partition_dict[partition][trial_name] = trial_length

        dataset_info['partition'] = partition_dict

        save_pkl_file(self.output_path, "dataset_info.pkl", dataset_info)

    def compact_labels(self):
        for partition, labels in self.annotation_to_partition_dict.items():
            if partition != "Test_Set":
                for label in tqdm(labels, total=len(labels)):
                    trial_name = label[:-4]
                    label_path = os.path.join(self.annotation_path, partition, label)
                    trial_length = txt_row_count(label_path)

                    npy_folder = os.path.join(self.output_path, "npy_data", trial_name)
                    os.makedirs(npy_folder, exist_ok=True)

                    npy_label_path = os.path.join(npy_folder, "label.npy")
                    if not os.path.isfile(npy_label_path):
                        label_data = pd.read_csv(label_path, sep=",").values
                        label_sampled = []
                        for i in range(trial_length):
                            if self.labeled_frame_indices_of_each_video[partition][trial_name][i] == 1:
                                label_sampled.append(label_data[i, :])

                        label_sampled = np.stack(label_sampled)
                        np.save(npy_label_path, label_sampled)

    def compact_aural_features(self):
        for partition, labels in self.annotation_to_partition_dict.items():
            for label in tqdm(labels, total=len(labels), desc=partition):
                trial_name = label[:-4]
                if trial_name.endswith("_left"):
                    trial_name_without_left_right = trial_name[:-5]
                elif trial_name.endswith("_right"):
                    trial_name_without_left_right = trial_name[:-6]
                else:
                    trial_name_without_left_right = trial_name

                label_path = os.path.join(self.annotation_path, partition, label)
                trial_length = txt_row_count(label_path)

                npy_folder = os.path.join(self.output_path, "npy_data", trial_name)
                os.makedirs(npy_folder, exist_ok=True)

                current_video_path = self.get_video_path(label, partition)
                video = cv2.VideoCapture(current_video_path)
                video_fps = video.get(cv2.CAP_PROP_FPS)

                # For mfcc feature
                npy_path_mfcc = os.path.join(npy_folder, "mfcc.npy")
                if not os.path.isfile(npy_path_mfcc):
                    mfcc_path = os.path.join(self.output_path, "audio_features_mfcc", partition,
                                             trial_name_without_left_right + ".csv")
                    mfcc_feature_matrix = pd.read_csv(mfcc_path, sep=";", usecols=range(2, 41)).values
                    mfcc_feature_sampled = []
                    for i in range(trial_length):
                        if self.labeled_frame_indices_of_each_video[partition][trial_name][i] == 1:
                            index_to_sample = int(round(i * 1 / video_fps / 0.01))
                            if index_to_sample >= len(mfcc_feature_matrix):
                                index_to_sample = -1
                            mfcc_feature_sampled.append(mfcc_feature_matrix[index_to_sample])

                    mfcc_feature_sampled = np.stack(mfcc_feature_sampled)

                    np.save(npy_path_mfcc, mfcc_feature_sampled)

                # For vggish feature
                npy_path_egemaps = os.path.join(npy_folder, "egemaps.npy")
                if not os.path.isfile(npy_path_egemaps):
                    egemaps_path = os.path.join(self.output_path, "audio_features_egemaps", partition,
                                                trial_name_without_left_right + ".csv")
                    egemaps_feature_matrix = pd.read_csv(egemaps_path, sep=";", usecols=range(2, 25)).values
                    egemaps_feature_sampled = []
                    for i in range(trial_length):
                        if self.labeled_frame_indices_of_each_video[partition][trial_name][i] == 1:
                            index_to_sample = int(round(i * 1 / video_fps / 0.01))
                            if index_to_sample >= len(egemaps_feature_matrix):
                                index_to_sample = -1

                            egemaps_feature_sampled.append(egemaps_feature_matrix[index_to_sample])

                    egemaps_feature_sampled = np.stack(egemaps_feature_sampled)

                    np.save(npy_path_egemaps, egemaps_feature_sampled)

                # For vggish feature
                npy_path_vggish = os.path.join(npy_folder, "vggish.npy")
                if not os.path.isfile(npy_path_vggish):
                    vggish_path = os.path.join(self.output_path, "audio_features_vggish", partition,
                                                trial_name_without_left_right + ".npy")
                    vggish_feature_matrix = np.load(vggish_path, mmap_mode="c")

                    vggish_feature_sampled = []
                    for i in range(trial_length):
                        if self.labeled_frame_indices_of_each_video[partition][trial_name][i] == 1:
                            index_to_sample = i
                            if index_to_sample >= len(vggish_feature_matrix):
                                index_to_sample = -1
                            vggish_feature_sampled.append(vggish_feature_matrix[index_to_sample, :])

                    vggish_feature_sampled = np.stack(vggish_feature_sampled)
                    np.save(npy_path_vggish, vggish_feature_sampled)

    def compact_visual_features(self):
        for partition, labels in self.annotation_to_partition_dict.items():
            for label in tqdm(labels, total=len(labels), desc=partition):
                if partition != "Test_Set":
                    trial_name = label[:-4]
                    label_path = os.path.join(self.annotation_path, partition, label)
                    trial_length = txt_row_count(label_path)
                else:
                    trial_name = label
                    corresponding_video = self.get_video_path(trial_name + ".txt", partition)
                    video = cv2.VideoCapture(corresponding_video)
                    trial_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

                input_path = os.path.join(self.image_path, trial_name)
                npy_folder = os.path.join(self.output_path, "npy_data", trial_name)
                os.makedirs(npy_folder, exist_ok=True)

                # For frame
                npy_frame_path = os.path.join(npy_folder, "frame.npy")
                frame_matrix = []
                black = np.zeros((48, 48, 3), dtype=np.int8)
                if not os.path.isfile(npy_frame_path):

                    for i in range(trial_length):
                        current_frame_path = os.path.join(input_path, str(i + 1).zfill(5) + ".jpg")
                        if self.labeled_frame_indices_of_each_video[partition][trial_name][i] == 1:
                            if os.path.isfile(current_frame_path):
                                current_frame = Image.open(current_frame_path)
                                current_frame = current_frame.resize((48, 48))
                                frame_matrix.append(np.array(current_frame))
                            else:
                                frame_matrix.append(black)

                    frame_matrix = np.stack(frame_matrix)
                    np.save(npy_frame_path, frame_matrix)

                # # For facial landmark
                # npy_flm_path = os.path.join(npy_folder, "landmark.npy")
                # landmark_matrix = []
                #
                # if not os.path.isfile(npy_flm_path):
                #     flm_path = os.path.join(self.output_path, "visual_features_openface_48", trial_name + ".csv")
                #     flm_data_x = pd.read_csv(flm_path, usecols=range(5, 73)).values
                #     flm_data_y = pd.read_csv(flm_path, usecols=range(73, 141)).values
                #     flm_data = np.concatenate((flm_data_x[...,None], flm_data_y[...,None]), axis=2)
                #     success_flm_indices = np.where(pd.read_csv(flm_path, usecols=range(4, 5)).values == 1)
                #     success_flm_data = flm_data[success_flm_indices[0]]
                #     success_flm_data = standardize_facial_landmarks(success_flm_data)
                #     flm_data[success_flm_indices[0]] = success_flm_data
                #
                #     for i in range(trial_length):
                #         if self.labeled_frame_indices_of_each_video[partition][trial_name][i] == 1:
                #             if i >= len(flm_data):
                #                 i = -1
                #             landmark_matrix.append(flm_data[i])
                #
                #     landmark_matrix = np.stack(landmark_matrix)
                #     np.save(npy_flm_path, landmark_matrix)
                #
                # # For facial action unit
                # npy_au_path = os.path.join(npy_folder, "au.npy")
                # au_matrix = []
                #
                # if not os.path.isfile(npy_au_path):
                #     au_path = os.path.join(self.output_path, "visual_features_openface_48", trial_name + ".csv")
                #     au_data = pd.read_csv(au_path, usecols=range(141, 158)).values
                #
                #     for i in range(trial_length):
                #         if self.labeled_frame_indices_of_each_video[partition][trial_name][i] == 1:
                #             if i >= len(au_data):
                #                 i = -1
                #             au_matrix.append(au_data[i])
                #
                #     au_matrix = np.stack(au_matrix)
                #     np.save(npy_au_path, au_matrix)

    def extract_visual_features(self):

        for partition, labels in self.annotation_to_partition_dict.items():
            for label in tqdm(labels, total=len(labels)):
                input_path = os.path.join(self.image_path, label[:-4])
                output_path = os.path.join(self.output_path, "visual_features_openface_48")
                os.makedirs(output_path, exist_ok=True)

                openface = OpenFaceController(openface_path=self.openface_config['openface_directory'],
                                              output_directory=output_path)
                _ = openface.process_video(
                    input_filename=input_path, output_filename=label[:-4], **self.openface_config)

    def get_available_frame_indices(self):
        available_frame_indices = {'Train_Set': {}, 'Validation_Set': {}, 'Test_Set': {}}
        intermediate_pkl_path = os.path.join(self.output_path, "available_frame_indices.pkl")
        if os.path.isfile(intermediate_pkl_path):
            with open(intermediate_pkl_path, 'rb') as handle:
                available_frame_indices = pickle.load(handle)
        else:
            # If this is not done before, then do it now.
            for partition, index_dict in self.labeled_frame_indices_of_each_video.items():

                for trial, index in tqdm(index_dict.items(), total=len(index_dict)):

                    available_indices = []
                    for i, flag in enumerate(index):
                        if flag == 1:
                            frame_id = str(i + 1).zfill(5)
                            frame_path = os.path.join(self.image_path, trial, frame_id + ".jpg")

                            if os.path.isfile(frame_path):
                                available_indices.append(1)
                            else:
                                available_indices.append(0)
                        else:
                            available_indices.append(-1)
                    available_frame_indices[partition].update({trial: available_indices})

            with open(intermediate_pkl_path, 'wb') as handle:
                pickle.dump(available_frame_indices, handle)

        return available_frame_indices

    def get_labeled_frame_indices(self):

        labeled_frame_indices = {'Train_Set': {}, 'Validation_Set': {}, 'Test_Set': {}}
        intermediate_pkl_path = os.path.join(self.output_path, "labeled_frame_indices.pkl")
        if os.path.isfile(intermediate_pkl_path):
            with open(intermediate_pkl_path, 'rb') as handle:
                labeled_frame_indices = pickle.load(handle)
        else:
            # If this is not done before, then do it now.
            for partition, labels in self.annotation_to_partition_dict.items():

                for label in tqdm(labels, total=len(labels)):

                    if partition != "Test_Set":
                        label_path = os.path.join(self.annotation_path, partition, label)
                        corresponding_video = self.get_video_path(label, partition)
                        # image_path = os.path.join(self.image_path, label[:-4])

                        # video = cv2.VideoCapture(corresponding_video)
                        # video_frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)

                        df = pd.read_csv(label_path, delimiter=",")
                        num_label_points, _ = df.shape

                        labeled_indices = []

                        for index, row in df.iterrows():
                            if row['valence'] != -5 and row['arousal'] != -5:
                                labeled_indices.append(1)
                            else:
                                labeled_indices.append(0)

                        labeled_frame_indices[partition].update({label[:-4]: labeled_indices})

                    else:
                        corresponding_video = self.get_video_path(label + ".txt", partition)
                        video = cv2.VideoCapture(corresponding_video)
                        video_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                        labeled_indices = np.ones(video_frame_count, dtype=np.int64).tolist()
                        labeled_frame_indices[partition].update({label: labeled_indices})

            with open(intermediate_pkl_path, 'wb') as handle:
                pickle.dump(labeled_frame_indices, handle)

        return labeled_frame_indices

    def get_video_path(self, label, partition):

        video_filename_without_extension = label[:-4]

        # Find the corresonding video filename.
        if video_filename_without_extension.endswith("_right"):
            video_filename_without_extension = video_filename_without_extension.split("_right")[0]
        elif video_filename_without_extension.endswith("_left"):
            video_filename_without_extension = video_filename_without_extension.split("_left")[0]

        if (video_filename_without_extension + ".avi") in self.video_to_partition_dict[partition]:
            corresponding_video_filename = video_filename_without_extension + ".avi"
        elif (video_filename_without_extension + ".mp4") in self.video_to_partition_dict[partition]:
            corresponding_video_filename = video_filename_without_extension + ".mp4"
        else:
            print(video_filename_without_extension)
            raise ValueError("Cannot find the corresponding video")

        corresponding_video = os.path.join(self.raw_video_path, corresponding_video_filename)

        return corresponding_video

    def extract_aural_features(self):

        for partition, wavs in self.video_to_partition_dict.items():
            for wav in tqdm(wavs, total=len(wavs)):

                input_path = os.path.join(self.output_path, "wav", partition, wav[:-4] + ".wav")
                corresponding_video = os.path.join(self.raw_video_path, wav)

                if "mfcc" in self.aural_feature_list:
                    feature = "mfcc"
                    opensmile_config_path = os.path.join(self.opensmile_config_path, feature, "MFCC12_0_D_A.conf")
                    opensmile_options = "-configfile " + opensmile_config_path + " -appendcsv 0 -timestampcsv 1 -headercsv 1 "
                    output_option = "-csvoutput"

                    output_folder = os.path.join(self.output_path, "audio_features_" + feature, partition)
                    os.makedirs(output_folder, exist_ok=True)
                    output_path = os.path.join(output_folder, wav[:-4] + ".csv")

                    command = "{opensmile_exe_path} {opensmile_options} -inputfile {input_path} {output_option} {output_path} -instname {filename} -output ?".format(
                        opensmile_exe_path=self.opensmile_exe_path, opensmile_options=opensmile_options,
                        input_path=input_path, output_option=output_option, output_path=output_path, filename=wav[:-4]
                    )

                    if not os.path.isfile(output_path):
                        subprocess.call(command)

                if "egemaps" in self.aural_feature_list:
                    feature = "egemaps"
                    opensmile_config_path = os.path.join(self.opensmile_config_path, feature, "v01a",
                                                         "eGeMAPSv01a.conf")
                    opensmile_options = "-configfile " + opensmile_config_path + " -appendcsvlld 0 -timestampcsvlld 1 -headercsvlld 1 "
                    output_option = "-lldcsvoutput"

                    output_folder = os.path.join(self.output_path, "audio_features_" + feature, partition)
                    os.makedirs(output_folder, exist_ok=True)
                    output_path = os.path.join(output_folder, wav[:-4] + ".csv")

                    command = "{opensmile_exe_path} {opensmile_options} -inputfile {input_path} {output_option} {output_path} -instname {filename} -output ?".format(
                        opensmile_exe_path=self.opensmile_exe_path, opensmile_options=opensmile_options,
                        input_path=input_path, output_option=output_option, output_path=output_path, filename=wav[:-4]
                    )

                    if not os.path.isfile(output_path):
                        subprocess.call(command)

                # if "vggish" in self.aural_feature_list:
                #     feature = "vggish"
                #
                #     output_folder = os.path.join(self.output_path, "audio_features_" + feature, partition)
                #     os.makedirs(output_folder, exist_ok=True)
                #     output_path = os.path.join(output_folder, wav[:-4] + ".npy")
                #
                #     if not os.path.isfile(output_path):
                #         video = cv2.VideoCapture(corresponding_video)
                #         video_fps = video.get(cv2.CAP_PROP_FPS)
                #         hop_sec = 1 / video_fps
                #
                #         vggish_feature = extract_vggish(wav_file=input_path, window_sec=0.96, hop_sec=hop_sec)
                #
                #         np.save(output_path, vggish_feature)

    def convert_video_to_wav(self):
        for partition, files in self.video_to_partition_dict.items():
            for file in tqdm(files, total=len(files)):
                # Input mp4 or avi
                input_path = os.path.join(self.raw_video_path, file)

                # Output wav
                output_folder = os.path.join(self.output_path, "wav", partition)
                os.makedirs(output_folder, exist_ok=True)
                output_path = os.path.join(output_folder, file[:-4] + ".wav")

                # ffmpeg command to execute
                # -ac 1 for mono, -ar 16000 for sample rate 16k, -q:v 0 for unchanging the quality.
                command = "ffmpeg -i {input_path} -ac 1 -ar 16000  -q:v 0 -f wav {output_path}".format(
                    input_path=input_path, output_path=output_path)

                # execute if the output does not exist
                if not os.path.isfile(output_path):
                    subprocess.call(command, shell=True)

    def generate_video_to_partition_dict(self):
        video_to_partition_dict = {'Train_Set': [], 'Validation_Set': [], 'Test_Set': []}

        # Training set and validation set.
        for root, dirs, files in os.walk(self.annotation_path):

            video_pool = os.listdir(self.raw_video_path)
            processed = []
            partition = root.split(sep=os.sep)[-1]
            for file in files:

                if file.endswith(".txt"):
                    if "right" in file:
                        file = file.split("_right")[0]
                    elif "left" in file:
                        file = file.split("_left")[0]
                    else:
                        file = file[:-4]

                    found_video = [video for video in video_pool if (file + ".mp4" == video or file + ".avi" == video)]

                    assert len(found_video) == 1

                    if found_video[0] not in processed:
                        processed.append(found_video[0])
                        video_to_partition_dict[partition].append(found_video[0])

        # Testing set
        video_pool = os.listdir(self.test_video_path)
        video_to_partition_dict['Test_Set'] = video_pool
        return video_to_partition_dict

    def generate_annotation_to_partition_dict(self):
        annotation_to_partition_dict = {'Train_Set': [], 'Validation_Set': [], 'Test_Set': []}

        for root, dirs, files in os.walk(self.annotation_path):

            # exclude the hidden file .DS_Store

            partition = root.split(sep=os.sep)[-1]
            for file in files:
                if file.endswith(".txt"):
                    # exclude the extension
                    annotation_to_partition_dict[partition].append(file)

        # For Test_Set:
        cropped_aligned_image_folder_pool = os.listdir(self.image_path)

        for partition, files in annotation_to_partition_dict.items():
            for file in files:
                if file[:-4] in cropped_aligned_image_folder_pool:
                    cropped_aligned_image_folder_pool.remove(file[:-4])
                else:
                    raise ValueError("Duplication found!")
        annotation_to_partition_dict['Test_Set'] = cropped_aligned_image_folder_pool
        return annotation_to_partition_dict


if __name__ == "__main__":
    a = ABAW2_Preprocessing()
