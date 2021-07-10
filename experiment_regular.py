import random

import numpy as np
import torch
import torch.backends.cudnn
import torch.cuda

from base.utils import detect_device, select_gpu, set_cpu_thread
from configs import config_processing as config
from model.model import my_2d1d, my_2dlstm, my_temporal
from base.dataset import ABAW2_VA_Arranger, ABAW2_VA_Dataset
from base.checkpointer import Checkpointer
from base.parameter_control import ParamControl
from base.trainer import ABAW2Trainer
from base.loss_function import CCCLoss
import os


class Experiment(object):
    def __init__(self, args):
        self.args = args
        self.experiment_name = args.experiment_name
        self.dataset_path = args.dataset_path
        self.model_load_path = args.model_load_path
        self.model_save_path = args.model_save_path
        self.resume = args.resume
        self.debug = args.debug
        self.config = config

        self.gpu = args.gpu
        self.cpu = args.cpu
        # If the code is to run on high-performance computer, which is usually not
        # available to specify gpu index and cpu threads, then set them to none.
        if self.args.high_performance_cluster:
            self.gpu = None
            self.cpu = None

        self.stamp = args.stamp

        self.head = "single-headed"
        if args.head == "mh":
            self.head = "multi-headed"
        elif args.head == "nh":
            self.head = "no-headed"

        self.train_emotion = args.train_emotion

        self.emotion_dimension = self.get_train_emotion(args.train_emotion, args.head)
        self.modality = args.modality

        self.backbone_state_dict = args.backbone_state_dict
        self.backbone_mode = args.backbone_mode

        self.input_dim = args.input_dim
        self.cnn1d_embedding_dim = args.cnn1d_embedding_dim
        self.cnn1d_channels = args.cnn1d_channels
        self.cnn1d_kernel_size = args.cnn1d_kernel_size
        self.cnn1d_dropout = args.cnn1d_dropout
        self.cnn1d_attention = args.cnn1d_attention
        self.lstm_embedding_dim = args.lstm_embedding_dim
        self.lstm_hidden_dim = args.lstm_hidden_dim
        self.lstm_dropout = args.lstm_dropout
        self.lstm_bidirectional = args.lstm_bidirectional

        self.adversarial = args.adversarial
        self.cross_validation = args.cross_validation
        self.folds_to_run = args.folds_to_run
        if not self.cross_validation:
            self.folds_to_run = [0]

        self.milestone = args.milestone
        self.learning_rate = args.learning_rate
        self.min_learning_rate = args.min_learning_rate
        self.early_stopping = args.early_stopping
        self.patience = args.patience
        self.time_delay = args.time_delay
        self.num_epochs = args.num_epochs
        self.min_num_epochs = args.min_num_epochs
        self.factor = args.factor

        self.window_length = args.window_length
        self.hop_length = args.hop_length
        self.batch_size = args.batch_size

        self.metrics = args.metrics
        self.release_count = args.release_count
        self.gradual_release = args.gradual_release
        self.load_best_at_each_epoch = args.load_best_at_each_epoch

        self.mlt_atten = args.mlt_attention
        self.pred_AU = args.pred_AU
        self.pred_Expre = args.pred_Expre

        self.save_plot = args.save_plot
        self.device = self.init_device()

        self.model_name = self.experiment_name + "_" + args.model_name + "_" + self.modality[
            0] + "_" + self.train_emotion + "_" + args.head + "_bs_" + str(self.batch_size) + "_lr_" + str(
            self.learning_rate) + "_mlr_" + str(self.min_learning_rate) + "_" + self.stamp

    def init_dataloader(self, fold):
        self.init_random_seed()

        arranger = ABAW2_VA_Arranger(self.dataset_path, window_length=self.window_length,
                                     hop_length=self.hop_length, debug=self.debug)

        # For fold = 0, it is the original partition.
        data_dict = arranger.resample_according_to_window_and_hop_length(fold)

        train_dataset = ABAW2_VA_Dataset(data_dict['Train_Set'], time_delay=self.time_delay, emotion=self.train_emotion,
                                         head=self.head, modality=self.modality, window_length=self.window_length,
                                         mode='train')
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=self.batch_size, shuffle=True)

        validate_dataset = ABAW2_VA_Dataset(data_dict['Validation_Set'], time_delay=self.time_delay,
                                            emotion=self.train_emotion, modality=self.modality,
                                            window_length=self.window_length,
                                            head=self.head, mode='validate')
        validate_loader = torch.utils.data.DataLoader(
            dataset=validate_dataset, batch_size=self.batch_size, shuffle=False)

        target_dataset = ABAW2_VA_Dataset(data_dict['Target_Set'], time_delay=self.time_delay,
                                          emotion=self.train_emotion,
                                          head=self.head, modality=self.modality, window_length=self.window_length,
                                          load_label=0,
                                          mode='train')

        target_loader = torch.utils.data.DataLoader(
            dataset=target_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        dataloader_dict = {'train': train_loader, 'validate': validate_loader, 'target': target_loader}
        return dataloader_dict

    def experiment(self):

        criterion = CCCLoss()

        for fold in iter(self.folds_to_run):

            save_path = os.path.join(self.model_save_path, self.model_name, str(fold))
            os.makedirs(save_path, exist_ok=True)

            checkpoint_filename = os.path.join(save_path, "checkpoint.pkl")

            dataloader_dict = self.init_dataloader(fold)
            model = self.init_model()

            trainer = ABAW2Trainer(model, model_name=self.model_name, learning_rate=self.learning_rate,
                                   min_learning_rate=self.min_learning_rate,
                                   metrics=self.metrics, save_path=save_path, early_stopping=self.early_stopping,
                                   train_emotion=self.train_emotion, patience=self.patience, factor=self.factor,
                                   emotional_dimension=self.emotion_dimension, head=self.head,
                                   max_epoch=self.num_epochs,
                                   load_best_at_each_epoch=self.load_best_at_each_epoch,
                                   window_length=self.window_length,
                                   milestone=self.milestone, criterion=criterion, verbose=True,
                                   save_plot=self.save_plot, fold=fold,
                                   device=self.device, pred_AU=self.pred_AU, pred_Expre=self.pred_Expre)

            parameter_controller = ParamControl(trainer, gradual_release=self.gradual_release,
                                                release_count=self.release_count, backbone_mode=self.backbone_mode)

            checkpoint_controller = Checkpointer(checkpoint_filename, trainer, parameter_controller, resume=self.resume)

            if self.resume:
                trainer, parameter_controller = checkpoint_controller.load_checkpoint()
            else:
                checkpoint_controller.init_csv_logger(self.args, self.config)

            if not trainer.fit_finished:
                trainer.fit(dataloader_dict, min_num_epochs=self.min_num_epochs, adversarial=self.adversarial,
                            save_model=True, parameter_controller=parameter_controller,
                            checkpoint_controller=checkpoint_controller)

    def init_model(self):
        self.init_random_seed()

        if self.head == "multi-headed":
            output_dim = 2
        elif self.head == "single-headed":
            output_dim = 1
        elif self.head == "no-headed":
            output_dim = 0

        if "2d1d" in self.model_name:
            model = my_2d1d(backbone_state_dict=self.backbone_state_dict, backbone_mode=self.backbone_mode,
                            embedding_dim=self.cnn1d_embedding_dim, channels=self.cnn1d_channels,
                            output_dim=output_dim, kernel_size=self.cnn1d_kernel_size, attention=self.cnn1d_attention,
                            adversarial=self.adversarial,
                            dropout=self.cnn1d_dropout, root_dir=self.model_load_path,
                            mlt_atten=self.mlt_atten, pred_AU=self.pred_AU, pred_Expre=self.pred_Expre)
            model.init()
        elif "2dlstm" in self.model_name:
            model = my_2dlstm(backbone_state_dict=self.backbone_state_dict, backbone_mode=self.backbone_mode,
                              embedding_dim=self.lstm_embedding_dim, hidden_dim=self.lstm_hidden_dim,
                              adversarial=self.adversarial,
                              output_dim=output_dim, dropout=self.lstm_dropout, bidirectional=self.lstm_bidirectional,
                              root_dir=self.model_load_path)
            model.init()
        elif "1d_only" in self.model_name or "lstm_only" in self.model_name:
            model = my_temporal(model_name=self.model_name, num_inputs=self.input_dim,
                                cnn1d_channels=self.cnn1d_channels, cnn1d_kernel_size=self.cnn1d_kernel_size,
                                cnn1d_dropout_rate=self.cnn1d_dropout, embedding_dim=self.lstm_embedding_dim,
                                bidirectional=self.lstm_bidirectional,
                                hidden_dim=self.lstm_hidden_dim, lstm_dropout_rate=self.lstm_dropout,
                                adversarial=self.adversarial,
                                output_dim=output_dim)
        else:
            raise ValueError("Unknown base_model!")

        return model

    @staticmethod
    def get_train_emotion(emotion_tag, head):

        emotion = ["Valence", "Arousal"]

        if emotion_tag == "arousal":
            if head == "sh":
                emotion = ["Arousal"]
        elif emotion_tag == "valence":
            if head == "sh":
                emotion = ["Valence"]

        return emotion

    @staticmethod
    def init_random_seed():
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.backends.cudnn.deterministic = True

    def init_device(self):
        device = detect_device()

        if not self.args.high_performance_cluster:
            select_gpu(self.gpu)
            set_cpu_thread(self.cpu)

        return device
