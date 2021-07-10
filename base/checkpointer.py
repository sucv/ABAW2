import os
import time

import numpy as np
import pandas as pd

from base.utils import save_pkl_file, load_single_pkl


class Checkpointer(object):
    r"""
    Save the trainer and parameter controller at runtime, and load them
        in another run if resume = 1.
    """
    def __init__(self, path, trainer, parameter_controller, resume):
        self.checkpoint = {}
        self.path = path
        self.trainer = trainer
        self.parameter_controller = parameter_controller
        self.resume = resume

        self.columns = []

    def save_log_to_csv(self, epoch, mean_train_record, mean_validate_record):
        num_layers_to_update = len(self.trainer.optimizer.param_groups[0]['params'])

        csv_records = [time.time(), epoch, int(self.trainer.best_epoch_info['epoch']), num_layers_to_update,
                       self.trainer.optimizer.param_groups[0]['lr'], self.trainer.train_losses[-1],
                       self.trainer.validate_losses[-1]]

        if self.trainer.head == "single-headed":
            if self.trainer.train_emotion == "arousal":
                csv_records.extend([
                    mean_train_record['Arousal']['rmse'][0], mean_train_record['Arousal']['pcc'][0][0],
                    mean_train_record['Arousal']['pcc'][0][1], mean_train_record['Arousal']['ccc'][0],
                    mean_validate_record['Arousal']['rmse'][0], mean_validate_record['Arousal']['pcc'][0][0],
                    mean_validate_record['Arousal']['pcc'][0][1], mean_validate_record['Arousal']['ccc'][0]])
            else:
                csv_records.extend([
                    mean_train_record['Valence']['rmse'][0], mean_train_record['Valence']['pcc'][0][0],
                    mean_train_record['Valence']['pcc'][0][1], mean_train_record['Valence']['ccc'][0],
                    mean_validate_record['Valence']['rmse'][0], mean_validate_record['Valence']['pcc'][0][0],
                    mean_validate_record['Valence']['pcc'][0][1], mean_validate_record['Valence']['ccc'][0]])
        else:
            csv_records.extend([
                                   mean_train_record['Arousal']['rmse'][0], mean_train_record['Arousal']['pcc'][0][0],
                                   mean_train_record['Arousal']['pcc'][0][1], mean_train_record['Arousal']['ccc'][0],
                                   mean_validate_record['Arousal']['rmse'][0], mean_validate_record['Arousal']['pcc'][0][0],
                                   mean_validate_record['Arousal']['pcc'][0][1], mean_validate_record['Arousal']['ccc'][0]] +
                               [
                                   mean_train_record['Valence']['rmse'][0], mean_train_record['Valence']['pcc'][0][0],
                                   mean_train_record['Valence']['pcc'][0][1], mean_train_record['Valence']['ccc'][0],
                                   mean_validate_record['Valence']['rmse'][0], mean_validate_record['Valence']['pcc'][0][0],
                                   mean_validate_record['Valence']['pcc'][0][1], mean_validate_record['Valence']['ccc'][0]])

        row_df = pd.DataFrame(data=csv_records)
        row_df.T.to_csv(self.trainer.csv_filename, mode='a', index=False, header=False)

    def init_csv_logger(self, args, config):

        self.trainer.csv_filename = os.path.join(self.trainer.save_path, "training_logs.csv")

        # Record the arguments.
        arguments_dict = vars(args)
        arguments_dict = pd.json_normalize(arguments_dict, sep='_')

        df_args = pd.DataFrame(data=arguments_dict)
        df_args.to_csv(self.trainer.csv_filename, index=False)

        config = pd.json_normalize(config, sep='_')
        df_config = pd.DataFrame(data=config)
        df_config.to_csv(self.trainer.csv_filename, mode='a', index=False)

        self.columns = ['time', 'epoch', 'best_epoch', 'layer_to_update', 'lr',
                        'tr_loss', 'val_loss']

        if self.trainer.head == "single-headed":
            if self.trainer.train_emotion == "arousal":
                self.columns.extend(['tr_rmse_a', 'tr_pcc_a_v', 'tr_pcc_a_conf', 'tr_ccc_a',
                                     'val_rmse_a', 'val_pcc_a_v', 'val_pcc_a_conf', 'val_ccc_a'])
            else:
                self.columns.extend(['tr_rmse_v', 'tr_pcc_v_v', 'tr_pcc_v_conf', 'tr_ccc_v',
                                     'val_rmse_v', 'val_pcc_v_v', 'val_pcc_v_conf', 'val_ccc_v'])
        else:
            self.columns.extend(['tr_rmse_a', 'tr_pcc_a_v', 'tr_pcc_a_conf', 'tr_ccc_a',
                                 'val_rmse_a', 'val_pcc_a_v', 'val_pcc_a_conf', 'val_ccc_a']
                                + ['tr_rmse_v', 'tr_pcc_v_v', 'tr_pcc_v_conf', 'tr_ccc_v',
                                   'val_rmse_v', 'val_pcc_v_v', 'val_pcc_v_conf', 'val_ccc_v'])

        df = pd.DataFrame(columns=self.columns)
        df.to_csv(self.trainer.csv_filename, mode='a', index=False)

    def load_checkpoint(self):
        # If checkpoint file exists, then read it.
        if os.path.isfile(self.path):
            print("Loading checkpoint. Are you sure it is intended?")
            self.checkpoint = {**self.checkpoint, **load_single_pkl(self.path)}
            print("Checkpoint loaded!")

            self.trainer = self.checkpoint['trainer']
            self.trainer.resume = True
            self.parameter_controller = self.checkpoint['param_control']
            self.parameter_controller.trainer = self.trainer
        else:
            raise ValueError("Checkpoint not exists!!")
        return self.trainer, self.parameter_controller

    def save_checkpoint(self, trainer, parameter_controller, path):
        self.checkpoint['trainer'] = trainer
        self.checkpoint['param_control'] = parameter_controller

        if path:
            print("Saving checkpoint.")
            save_pkl_file(path, "checkpoint.pkl", self.checkpoint)
            print("Checkpoint saved.")