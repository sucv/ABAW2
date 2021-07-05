import os
import subprocess
import glob

import numpy as np
import pickle


class OpenFaceController(object):
    def __init__(self, openface_path, output_directory):
        self.openface_path = openface_path
        self.output_directory = output_directory

    def get_openface_command(self, **kwargs):
        openface_path = self.openface_path
        input_flag = kwargs['input_flag']
        output_features = kwargs['output_features']
        output_action_unit = kwargs['output_action_unit']
        output_image_flag = kwargs['output_image_flag']
        output_image_size = kwargs['output_image_size']
        output_image_format = kwargs['output_image_format']
        output_filename_flag = kwargs['output_filename_flag']
        output_directory_flag = kwargs['output_directory_flag']
        output_directory = self.output_directory
        output_image_mask_flag = kwargs['output_image_mask_flag']

        command = openface_path + input_flag + " {input_filename} " + output_features \
                  + output_action_unit + output_image_flag + output_image_size \
                  + output_image_format + output_filename_flag + " {output_filename} " \
                  + output_directory_flag + output_directory + output_image_mask_flag
        return command

    def process_video(self, input_filename, output_filename, **kwargs):

        # Quote the file name if spaces occurred.
        if " " in input_filename:
            input_filename = '"' + input_filename + '"'

        command = self.get_openface_command(**kwargs)
        command = command.format(
            input_filename=input_filename, output_filename=output_filename)

        if not os.path.isfile(os.path.join(self.output_directory, output_filename + ".csv")):
            subprocess.call(command, shell=True)

        return os.path.join(self.output_directory, output_filename)


def get_filename_from_a_folder_given_extension(folder, extension):
    file_list = []
    import os
    for root, dirs, files in os.walk(folder):
        if not root.startswith("."):
            partition = root.split(sep=os.sep)[-1]
            for file in files:
                if file.endswith(extension):
                    print(os.path.join(root, file))


def txt_row_count(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i


def standardize_facial_landmarks(landmark_sequence):
    from sklearn import preprocessing

    landmark_sequence_scaled = []
    for i in range(landmark_sequence.shape[2]):
        current_dim = landmark_sequence[:, :, i]
        scaler = preprocessing.StandardScaler().fit(current_dim)
        landmark_sequence_scaled.append(scaler.transform(current_dim))
    landmark_sequence_scaled = np.transpose(np.stack(landmark_sequence_scaled), (1, 2, 0))
    return landmark_sequence_scaled


def load_single_pkl(directory, filename=None, extension='.pkl'):
    r"""
    Load one pkl file according to the filename.
    """
    if filename is not None:
        fullname = os.path.join(directory, filename + extension)
    else:
        fullname = directory if directory.endswith(".pkl") else directory + ".pkl"

    fullname = glob.glob(fullname)[0]

    with open(fullname, 'rb') as f:
        pkl_file = pickle.load(f)

    return pkl_file


def save_pkl_file(directory, filename, data):
    os.makedirs(directory, exist_ok=True)
    fullname = os.path.join(directory, filename)

    with open(fullname, 'wb') as pkl_file:
        pickle.dump(data, pkl_file)

def detect_device():
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cuda:1')
    return device


def select_gpu(index):
    import torch
    r"""
    Choose which gpu to use.
    :param index: (int), the index corresponding to the desired gpu. For example,
        0 means the 1st gpu.
    """
    torch.cuda.set_device(index)


def set_cpu_thread(number):
    import torch
    r"""
    Set the maximum thread of cpu for torch module.
    :param number: (int), the number of thread allowed, usually 1 is enough.
    """
    torch.set_num_threads(number)
