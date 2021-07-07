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

        # Let the 0-th fold be the original partition.
        if fold != 0:
            random.shuffle(trials_of_train_set)

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
    def __init__(self, data_list, time_delay=0, emotion="both", head="multi_head", modality=['frame'],
                 window_length=300, load_label=10, mode='train', au_n=12):
        self.data_list = data_list
        self.time_delay = time_delay
        self.emotion = emotion
        self.head = head
        self.modality = modality
        self.window_length = window_length
        self.load_label = load_label
        self.mode = mode
        self.au_n = au_n
        self.get_3D_transforms()
        self.prepare_labels()

    def get_3D_transforms(self):
        normalize = GroupNormalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        self.image_transforms = {}

        if self.mode == 'train':

            self.image_transforms["frame"] = transforms.Compose([
                GroupNumpyToPILImage(0),
                GroupRandomCrop(48, 40),
                GroupRandomHorizontalFlip(),
                Stack(),
                ToTorchFormatTensor(),
                normalize
            ])

        if self.mode != 'train':
            self.image_transforms["frame"] = transforms.Compose([
                GroupNumpyToPILImage(0),
                GroupCenterCrop(40),
                Stack(),
                ToTorchFormatTensor(),
                normalize
            ])
        for m in self.modality:
            if m == 'frame': continue
            self.image_transforms[m] = transforms.Compose([])

    @staticmethod
    def load_data(directory, indices, filename):
        filename = os.path.join(directory, filename)
        frames = np.load(filename, mmap_mode='c')[indices]
        return frames

    def prepare_labels(self):
        self.label_list = []
        self.AU_label_list = []
        self.Expre_label_list = []
        for path, trial, indices, length in self.data_list:
            if self.load_label:
                if length < self.window_length:
                    labels = np.zeros((self.window_length, 2), dtype=np.float32)
                    labels[indices] = self.load_data(path, indices, "label.npy")
                else:
                    labels = self.load_data(path, indices, "label.npy")
            else:
                labels = 0
            self.label_list.append(labels)
            AU_path = os.path.join(path, "AU_label.npy")
            if os.path.exists(AU_path):
                if length < self.window_length:
                    temp = np.zeros((self.window_length, self.au_n), dtype=np.float32)
                    temp[indices] = self.load_data(path, indices, "AU_label.npy")
                else:
                    temp = self.load_data(path, indices, "AU_label.npy")
            else:
                temp = np.stack([np.array([-1 for _ in range(self.au_n)]) for _ in range(self.window_length)])
            self.AU_label_list.append([temp, temp != -1])
            Expre_path = os.path.join(path, "EXPRE_label.npy")
            if os.path.exists(Expre_path):
                if length < self.window_length:
                    temp = np.zeros((self.window_length, 1), dtype=np.float32)
                    temp[indices] = self.load_data(path, indices, "EXPRE_label.npy")
                else:
                    temp = self.load_data(path, indices, "EXPRE_label.npy")
            else:
                temp = np.stack([np.array([-1]) for _ in range(self.window_length)])
            self.Expre_label_list.append([temp, temp != -1])

    def __getitem__(self, index):
        path = self.data_list[index][0]
        trial = self.data_list[index][1]
        indices = self.data_list[index][2]
        length = self.data_list[index][3]
        labels = self.label_list[index]

        output = {'frame': None, 'AUs': self.AU_label_list[index], 'Expres': self.Expre_label_list[index]}
        features = []
        for m in self.modality:
            if m == 'frame':
                if length < self.window_length:
                    data = np.zeros((self.window_length, 48, 48, 3), dtype=np.int16)
                    data[indices] = self.load_data(path, indices, "frame.npy")
                else:
                    data = self.load_data(path, indices, "frame.npy")
                output['frame'] = self.image_transforms[m](data)
                continue
            elif m == 'landmark':
                data = self.load_data(path, indices, "%s.npy" % m)
                data = np.array(data.reshape(len(data), -1), dtype=np.float32)
            else:
                data = self.load_data(path, indices, "%s.npy" % m)
            features.append(self.image_transforms[m](data))
        output['feature'] = torch.cat(features) if len(features) != 0 else 0
        if self.load_label:
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

        return output, labels, trial, length, indices

    def __len__(self):
        return len(self.data_list)

