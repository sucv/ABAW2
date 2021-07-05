from torch.utils.data import Dataset

import os
from base.utils import load_single_pkl
from base.transforms3D import *

from torchvision.transforms import transforms
import numpy as np
import random
from operator import itemgetter


class ABAW2_VA_Arranger(object):
    def __init__(self, dataset_path, window_length=300, hop_length=300, debug=0):
        self.dataset_path = os.path.join(dataset_path, "npy_data")
        self.dataset_info = load_single_pkl(directory=dataset_path, filename="dataset_info")
        self.mean_std_info = load_single_pkl(directory=dataset_path, filename="mean_std_dict")
        self.window_length = window_length
        self.hop_length = hop_length
        self.debug = debug

    @staticmethod
    def generate_partition_dict_for_cross_validation(partition_dict, fold):
        new_partition_dict = {'Train_Set': {}, 'Validation_Set': {}, 'Target_Set': {}}
        partition_pool = {**partition_dict['Train_Set'], **partition_dict['Validation_Set']}

        trials_of_train_set = list(partition_dict['Train_Set'].keys())
        trials_of_original_validate_set = list(partition_dict['Validation_Set'].keys())
        trials_of_putative_test_set = list(partition_dict['Target_Set'].keys())

        fold_0_trials = trials_of_train_set[slice(0, 70)]
        fold_1_trials = trials_of_train_set[slice(70, 140)]
        fold_2_trials = trials_of_train_set[slice(140, 210)]
        fold_3_trials = trials_of_train_set[slice(210, 280)]
        fold_4_trials = trials_of_train_set[slice(280, 351)]
        fold_5_trials = trials_of_original_validate_set

        fold_n_trials = [
            fold_0_trials,
            fold_1_trials,
            fold_2_trials,
            fold_3_trials,
            fold_4_trials,
            fold_5_trials
        ]

        fold_index = np.roll(np.arange(len(fold_n_trials)), fold)
        ordered_trials = list(itemgetter(*fold_index)(fold_n_trials))

        for nth_fold, trials_of_a_fold in enumerate(ordered_trials):
            for trial in trials_of_a_fold:
                partition = "Train_Set"
                if nth_fold == len(fold_n_trials) - 1:
                    partition = "Validation_Set"
                new_partition_dict[partition].update({trial: partition_pool[trial]})

        new_partition_dict['Target_Set'] = partition_dict['Target_Set']
        return new_partition_dict

    def resample_according_to_window_and_hop_length(self, fold):
        partition_dict = self.dataset_info['partition']

        partition_dict = self.generate_partition_dict_for_cross_validation(partition_dict, fold)
        sampled_list = {'Train_Set': [], 'Validation_Set': [], 'Target_Set': []}

        for partition, trials in partition_dict.items():
            trial_count = 0
            for trial, length in trials.items():

                start = 0
                end = start + self.window_length

                if end < length:
                    # Windows before the last one
                    while end < length:
                        indices = np.arange(start, end)
                        path = os.path.join(self.dataset_path, trial)
                        sampled_list[partition].append([path, trial, indices, length])
                        start = start + self.hop_length
                        end = start + self.window_length

                    # The last window ends with the trial length, which ensures that no data are wasted.
                    start = length - self.window_length
                    end = length
                    indices = np.arange(start, end)
                    path = os.path.join(self.dataset_path, trial)
                    sampled_list[partition].append([path, trial, indices, length])
                else:
                    end = length
                    indices = np.arange(start, end)
                    path = os.path.join(self.dataset_path, trial)
                    sampled_list[partition].append([path, trial, indices, length])

                trial_count += 1

                if self.debug and trial_count >= self.debug:
                    break

        return sampled_list


class ABAW2_VA_Dataset(Dataset):
    def __init__(self, data_list, time_delay=0, emotion="both", head="multi_head", modality = ['frame'], window_length=300, mode='train', fold=0, mean_std_info=None):
        self.data_list = data_list
        self.time_delay = time_delay
        self.emotion = emotion
        self.head = head
        self.modality = modality
        self.window_length = window_length
        self.mode = mode
        self.partition = self.get_partition_name_from_mode()
        self.fold = fold
        self.mean_std_info = mean_std_info
        self.get_3D_transforms()

    def get_partition_name_from_mode(self):
        if self.mode == "train":
            partition = "Train_Set"
        elif self.mode == "validate":
            partition = "Validation_Set"
        elif self.mode == "test":
            partition = "Target_Set"
        else:
            raise ValueError("Unknown partition!")

        return partition

    def get_3D_transforms(self):
        normalize = GroupNormalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        if self.mode == 'train':
            if "frame" in self.modality:
                self.image_transforms = transforms.Compose([
                    GroupNumpyToPILImage(0),
                    GroupRandomCrop(48, 40),
                    GroupRandomHorizontalFlip(),
                    Stack(),
                    ToTorchFormatTensor(),
                    normalize
                ])

        if self.mode != 'train':
            if "frame" in self.modality:
                self.image_transforms = transforms.Compose([
                    GroupNumpyToPILImage(0),
                    GroupCenterCrop(40),
                    Stack(),
                    ToTorchFormatTensor(),
                    normalize
                ])

        if "mfcc" in self.modality:
            self.mfcc_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean_std_info["mfcc"][self.fold][self.partition]['mean'],
                                     std=self.mean_std_info["mfcc"][self.fold][self.partition]['std']),
            ])

        if "vggish" in self.modality:
            self.vggish_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean_std_info["vggish"][self.fold][self.partition]['mean'],
                                     std=self.mean_std_info["vggish"][self.fold][self.partition]['std']),
            ])



    @staticmethod
    def load_data(directory, indices, filename):
        filename = os.path.join(directory, filename)
        frames = np.load(filename, mmap_mode='c')[indices]
        return frames

    def __getitem__(self, index):
        path = self.data_list[index][0]
        trial = self.data_list[index][1]
        indices = self.data_list[index][2]
        length = self.data_list[index][3]
        labels = 0
        features = {}

        if "frame" in self.modality:

            if length < self.window_length:
                frames = np.zeros((self.window_length, 48, 48, 3), dtype=np.int16)
                frames[indices] = self.load_data(path, indices, "frame.npy")
            else:
                frames = self.load_data(path, indices, "frame.npy")

            frames = self.image_transforms(frames)
            features.update({'frame': frames})

        if "fau" in self.modality:

            if length < self.window_length:
                au = np.zeros((self.window_length, 17), dtype=np.float32)
                au[indices] = self.load_data(path, indices, "au.npy")
            else:
                au = self.load_data(path, indices, "au.npy")
            features.update({'au': au})

        if "flm" in self.modality:
            if length < self.window_length:
                landmark = np.zeros((self.window_length, 68, 2), dtype=np.float32)
                landmark[indices] = self.load_data(path, indices, "landmark.npy")
            else:
                landmark = self.load_data(path, indices, "landmark.npy")
            landmark = np.array(landmark.reshape(len(landmark), -1), dtype=np.float32)
            features.update({'landmark': landmark})

        if "mfcc" in self.modality:
            if length < self.window_length:
                mfcc = np.zeros((self.window_length, 39), dtype=np.float32)
                mfcc[indices] = self.load_data(path, indices, "mfcc.npy")
            else:
                mfcc = self.load_data(path, indices, "mfcc.npy").astype(np.float32)
            mfcc = self.mfcc_transforms(mfcc)
            features.update({'mfcc': mfcc})

        if "egemaps" in self.modality:
            if length < self.window_length:
                egemaps = np.zeros((self.window_length, 23), dtype=np.float32)
                egemaps[indices] = self.load_data(path, indices, "egemaps.npy")
            else:
                egemaps = self.load_data(path, indices, "egemaps.npy")
            features.update({'egemaps': egemaps})

        if "vggish" in self.modality:
            if length < self.window_length:
                vggish = np.zeros((self.window_length, 128), dtype=np.float32)
                vggish[indices] = self.load_data(path, indices, "vggish.npy")
            else:
                vggish = self.load_data(path, indices, "vggish.npy").astype(np.float32)
            # vggish = (vggish - (-0.105)) / 0.4476
            vggish = self.vggish_transforms(vggish)
            features.update({'vggish': vggish})

        if self.mode != "test":
            if length < self.window_length:
                labels = np.zeros((self.window_length, 2), dtype=np.float32)
                labels[indices] = self.load_data(path, indices, "label.npy")
            else:
                labels = self.load_data(path, indices, "label.npy")

            if self.head == "single-headed":
                if self.emotion == "arousal": #Arousal
                    labels = labels[:, 1][:, np.newaxis]
                elif self.emotion == "valence": # Valence
                    labels = labels[:, 0][:, np.newaxis]
                else:
                    raise ValueError("Unsupported emotional dimension for continuous labels!")

            # Deal with the label delay by making the time_delay-th label point as the 1st point and shift other points accordingly.
            # The previous last point is copied for time_delay times.
            labels = np.concatenate(
                (labels[self.time_delay:, :],
                 np.repeat(labels[-1, :][np.newaxis], repeats=self.time_delay, axis=0)), axis=0)

        if len(indices) < self.window_length:
            indices = np.arange(self.window_length)

        return features, labels, trial, length, indices

    def __len__(self):
        return len(self.data_list)

