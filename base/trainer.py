import os
import time
import copy
from tqdm import tqdm


import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.nn import MSELoss
import torch.utils.data

from base.logger import ContinuousOutputHandlerNPYTrial, ContinuousMetricsCalculatorTrial, ConcordanceCorrelationCoefficient, PlotHandlerTrial


class ABAW2Trainer(object):
    def __init__(self, model, model_name='2d1d', save_path=None, train_emotion='both', head='multi-headed', factor=0.1,
                 early_stopping=100, criterion=None, milestone=[0], patience=10, learning_rate=0.00001, device='cpu', num_classes=2, max_epoch=50, min_learning_rate=1e-7,
                 emotional_dimension=None, metrics=None, verbose=False, print_training_metric=False, save_plot=False, window_length=300,
                 load_best_at_each_epoch=False, fold=0, **kwargs):

        self.device = device
        self.model = model.to(device)
        self.model_name = model_name
        self.save_path = save_path
        self.fold = fold

        self.window_length = window_length
        self.num_classes = num_classes
        self.max_epoch = max_epoch
        self.start_epoch = 0
        self.early_stopping = early_stopping
        self.early_stopping_counter = self.early_stopping
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.patience = patience
        self.criterion = criterion
        self.factor = factor
        self.init_optimizer_and_scheduler()

        self.verbose = verbose

        self.device = device

        # Whether to show the information strings.
        self.verbose = verbose
        self.save_plot = save_plot

        # Whether print the metrics for training.
        self.print_training_metric = print_training_metric

        # What emotional dimensions to consider.
        self.emotional_dimension = emotional_dimension
        self.train_emotion = train_emotion
        self.head = head

        self.metrics = metrics

        # The learning rate, and the patience of schedule.
        self.learning_rate = learning_rate
        self.patience = patience

        # The networks.
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.model = model.to(device)

        self.init_optimizer_and_scheduler()

        # parameter_control
        self.milestone = milestone

        # For checkpoint
        self.fit_finished = False
        self.fold_finished = False
        self.resume = False
        self.time_fit_start = None
        self.combined_record_dict = {'train': {}, 'validate': {}}
        self.train_losses = []
        self.validate_losses = []
        self.csv_filename = None
        self.best_epoch_info = None
        self.load_best_at_each_epoch = load_best_at_each_epoch

    def init_optimizer_and_scheduler(self):
        if len(self.get_parameters()) != 0:
            self.optimizer = optim.Adam(self.get_parameters(), lr=self.learning_rate, weight_decay=0.001)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=self.patience,
                                                                        factor=self.factor)

    def get_parameters(self):
        r"""
        Get the parameters to update.
        :return:
        """
        params_to_update = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
        return params_to_update

    def train(self, data_loader, epoch):
        self.model.train()
        loss, result_dict = self.loop(data_loader, epoch,
                                      train_mode=True)
        return loss, result_dict

    def validate(self, data_loader, epoch):
        with torch.no_grad():
            self.model.eval()
            loss, result_dict = self.loop(data_loader, epoch, train_mode=False)
        return loss, result_dict

    def test(
            self,
            data_to_load,
            output_save_path
    ):
        if self.verbose:
            print("------")
            print("Starting testing, on device:", self.device)

        with torch.no_grad():
            self.model.eval()
            self.loop_test(data_to_load, output_save_path)


    def fit(
            self,
            data_to_load,
            num_epochs=100,
            min_num_epochs=10,
            checkpoint_controller=None,
            parameter_controller=None,
            save_model=False
    ):

        if self.verbose:
            print("------")
            print("Starting training, on device:", self.device)

        self.time_fit_start = time.time()
        start_epoch = self.start_epoch

        if self.best_epoch_info is None:
            self.best_epoch_info = {
                'model_weights': copy.deepcopy(self.model.state_dict()),
                'loss': 1e10,
                'ccc': -1e10
            }

        # Loop the epochs
        for epoch in np.arange(start_epoch, num_epochs):

            if self.fit_finished:
                if self.verbose:
                    print("\nEarly Stop!\n")
                break

            if epoch in self.milestone or parameter_controller.get_current_lr() < self.min_learning_rate:
                parameter_controller.release_param()
                self.model.load_state_dict(self.best_epoch_info['model_weights'])

                if parameter_controller.early_stop:
                    break

            time_epoch_start = time.time()

            print("There are {} layers to update.".format(len(self.optimizer.param_groups[0]['params'])))

            # Get the losses and the record dictionaries for training and validation.
            train_loss, train_record_dict = self.train(data_to_load['train'], epoch)

            # Combine the record to a long array for each subject.
            self.combined_record_dict['train'] = self.combine_record_dict(
                self.combined_record_dict['train'], train_record_dict)

            validate_loss, validate_record_dict = self.validate(data_to_load['validate'], epoch)

            self.combined_record_dict['validate'] = self.combine_record_dict(
                self.combined_record_dict['validate'], validate_record_dict)

            self.train_losses.append(train_loss)
            self.validate_losses.append(validate_loss)

            improvement = False

            if self.train_emotion == "both":
                validate_ccc = np.mean(
                    [validate_record_dict['overall'][emotion]['ccc'] for emotion in self.emotional_dimension])
            elif self.train_emotion == "arousal":
                validate_ccc = np.mean(validate_record_dict['overall']['Arousal']['ccc'])
            elif self.train_emotion == "valence":
                validate_ccc = np.mean(validate_record_dict['overall']['Valence']['ccc'])
            else:
                raise ValueError("Unknown emotion dimension!")

            # If a lower validate loss appears.
            if validate_ccc > self.best_epoch_info['ccc']:
                if save_model:
                    torch.save(self.model.state_dict(), os.path.join(self.save_path, "model_state_dict.pth"))

                improvement = True
                self.best_epoch_info = {
                    'model_weights': copy.deepcopy(self.model.state_dict()),
                    'loss': validate_loss,
                    'ccc': validate_ccc,
                    'epoch': epoch,
                    'scalar_metrics': {
                        'train_loss': train_loss,
                        'validate_loss': validate_loss,
                    },
                    'array_metrics': {
                        'train_metric_record': train_record_dict['overall'],
                        'validate_metric_record': validate_record_dict['overall']
                    }
                }

            if validate_loss < 0:
                print('validate loss negative')

            if self.verbose:
                print(
                    "\n Fold {:2} Epoch {:2} in {:.0f}s || Train loss={:.3f} | Val loss={:.3f} | LR={:.1e} | Release_count={} | best={} | "
                    "improvement={}-{}".format(
                        self.fold,
                        epoch + 1,
                        time.time() - time_epoch_start,
                        train_loss,
                        validate_loss,
                        self.optimizer.param_groups[0]['lr'],
                        parameter_controller.release_count,
                        int(self.best_epoch_info['epoch']) + 1,
                        improvement,
                        self.early_stopping_counter))

                print(train_record_dict['overall'])
                print(validate_record_dict['overall'])
                print("------")

            checkpoint_controller.save_log_to_csv(
                epoch, train_record_dict['overall'], validate_record_dict['overall'])

            # Early stopping controller.
            if self.early_stopping and epoch > min_num_epochs:
                if improvement:
                    self.early_stopping_counter = self.early_stopping
                else:
                    self.early_stopping_counter -= 1

                if self.early_stopping_counter <= 0:
                    self.fit_finished = True

            self.scheduler.step(validate_ccc)
            self.start_epoch = epoch + 1

            if self.load_best_at_each_epoch:
                self.model.load_state_dict(self.best_epoch_info['model_weights'])

            checkpoint_controller.save_checkpoint(self, parameter_controller, self.save_path)

        self.fit_finished = True
        checkpoint_controller.save_checkpoint(self, parameter_controller, self.save_path)

        self.model.load_state_dict(self.best_epoch_info['model_weights'])

    def loop(self, data_loader, epoch, train_mode=True):
        running_loss = 0.0

        output_handler = ContinuousOutputHandlerNPYTrial(self.emotional_dimension)
        continuous_label_handler = ContinuousOutputHandlerNPYTrial(self.emotional_dimension)

        # This object calculate the metrics, usually by root mean square error, pearson correlation
        # coefficient, and concordance correlation coefficient.
        metric_handler = ContinuousMetricsCalculatorTrial(self.metrics, self.emotional_dimension,
                                                          output_handler, continuous_label_handler)
        total_batch_counter = 0
        for batch_index, (X, Y, trials, lengths, indices) in tqdm(enumerate(data_loader), total=len(data_loader)):
            total_batch_counter += len(trials)

            if 'frame' in X:
                inputs = X['frame'].to(self.device)

            if 'landmark' in X:
                inputs1 = X['landmark'].to(self.device)

            if 'au' in X:
                inputs2 = X['au'].to(self.device)

            if 'mfcc' in X:
                inputs3 = X['mfcc'].to(self.device)

            if 'egemaps' in X:
                inputs = X['egemaps'].to(self.device)

            if 'vggish' in X:
                inputs4 = X['vggish'].to(self.device)

            labels = Y.float().to(self.device)

            # Determine the weight for loss function
            loss_weights = torch.ones_like(labels).to(self.device)
            if train_mode:
                self.optimizer.zero_grad()

                if self.head == "multi-headed":
                    if self.train_emotion == "both":
                        loss_weights[:, :, 0] *= 1
                        loss_weights[:, :, 1] *= 1
                    elif self.train_emotion == "arousal":
                        loss_weights[:, :, 0] *= 1
                        loss_weights[:, :, 1] *= 10
                    elif self.train_emotion == "valence":
                        loss_weights[:, :, 0] *= 10
                        loss_weights[:, :, 1] *= 1
                    else:
                        raise ValueError("Unknown emotion dimention to train!")

            if len(X.keys())>1:
                outputs = self.model(inputs, inputs4, inputs3)
            else:
                outputs = self.model(inputs)

            output_handler.update_output_for_seen_trials(outputs.detach().cpu().numpy(), trials, indices, lengths)
            continuous_label_handler.update_output_for_seen_trials(labels.detach().cpu().numpy(), trials, indices, lengths)

            loss = self.criterion(outputs, labels, loss_weights)

            running_loss += loss.mean().item()

            if train_mode:
                loss.backward()
                self.optimizer.step()

            #  print_progress(batch_index, len(data_loader))

        epoch_loss = running_loss / total_batch_counter

        output_handler.average_trial_wise_records()
        continuous_label_handler.average_trial_wise_records()

        output_handler.concat_records()
        continuous_label_handler.concat_records()

        # Compute the root mean square error, pearson correlation coefficient and significance, and the
        # concordance correlation coefficient.
        # They are calculated by  first concatenating all the output
        # and continuous labels to two long arrays, and then calculate the metrics.
        metric_handler.calculate_metrics()
        epoch_result_dict = metric_handler.metric_record_dict

        if self.save_plot:
            # This object plot the figures and save them.
            plot_handler = PlotHandlerTrial(self.metrics, self.emotional_dimension, epoch_result_dict,
                                            output_handler.trialwise_records,
                                            continuous_label_handler.trialwise_records,
                                            epoch=epoch, train_mode=train_mode,
                                            directory_to_save_plot=self.save_path)
            plot_handler.save_output_vs_continuous_label_plot()

        return epoch_loss, epoch_result_dict

    def loop_test(self, data_loader, output_save_path):

        output_handler = ContinuousOutputHandlerNPYTrial(self.emotional_dimension)
        continuous_label_handler = ContinuousOutputHandlerNPYTrial(self.emotional_dimension)

        # This object calculate the metrics, usually by root mean square error, pearson correlation
        # coefficient, and concordance correlation coefficient.
        metric_handler = ContinuousMetricsCalculatorTrial(self.metrics, self.emotional_dimension,
                                                          output_handler, continuous_label_handler)

        total_batch_counter = 0
        for batch_index, (X, Y, trials, lengths, indices) in tqdm(enumerate(data_loader), total=len(data_loader)):
            total_batch_counter += len(trials)

            if 'frame' in X:
                inputs = X['frame'].to(self.device)

            if 'landmark' in X:
                inputs1 = X['landmark'].to(self.device)

            if 'au' in X:
                inputs2 = X['au'].to(self.device)

            if 'mfcc' in X:
                inputs3 = X['mfcc'].to(self.device)

            if 'egemaps' in X:
                inputs = X['egemaps'].to(self.device)

            if 'vggish' in X:
                inputs4 = X['vggish'].to(self.device)

            labels = Y.float().to(self.device)

            if len(X.keys())>1:
                outputs = self.model(inputs, inputs4, inputs3)
            else:
                outputs = self.model(inputs)

            output_handler.update_output_for_seen_trials(outputs.detach().cpu().numpy(), trials, indices, lengths)
            continuous_label_handler.update_output_for_seen_trials(labels.detach().cpu().numpy(), trials, indices,
                                                                   lengths)

        output_handler.average_trial_wise_records()
        continuous_label_handler.average_trial_wise_records()

        output_handler.concat_records()
        continuous_label_handler.concat_records()

        # Compute the root mean square error, pearson correlation coefficient and significance, and the
        # concordance correlation coefficient.
        # They are calculated by  first concatenating all the output
        # and continuous labels to two long arrays, and then calculate the metrics.
        metric_handler.calculate_metrics()
        epoch_result_dict = metric_handler.metric_record_dict

        plot_handler = PlotHandlerTrial(self.metrics, self.emotional_dimension, epoch_result_dict,
                                        output_handler.trialwise_records,
                                        continuous_label_handler.trialwise_records,
                                        epoch='final', train_mode=False,
                                        directory_to_save_plot=self.save_path)
        plot_handler.save_output_vs_continuous_label_plot()

        for trial, record in output_handler.trialwise_records.items():
            for emotion, pred in record.items():
                result = pd.DataFrame({emotion: pred})
                result_path = os.path.join(output_save_path, emotion)
                os.makedirs(result_path, exist_ok=True)
                result_path = os.path.join(result_path, trial + ".txt")
                result.to_csv(result_path, sep=',', index=None)

        print(0)


    def combine_record_dict(self, main_record_dict, epoch_record_dict):
        r"""
        Append the metric recording dictionary of an epoch to a main record dictionary.
            Each single term from epoch_record_dict will be appended to the corresponding
            list in min_record_dict.
        Therefore, the minimum terms in main_record_dict are lists, whose element number
            are the epoch number.
        """

        # If the main record dictionary is blank, then initialize it by directly copying from epoch_record_dict.
        # Since the minimum term in epoch_record_dict is list, it is available to append further.
        if not bool(main_record_dict):
            main_record_dict = epoch_record_dict
            return main_record_dict

        # Iterate the dict and append each terms from epoch_record_dict to
        # main_record_dict.
        for (trial, main_subject_record), (_, epoch_subject_record) \
                in zip(main_record_dict.items(), epoch_record_dict.items()):

            # Go through emotions, e.g., Arousal and Valence.
            for emotion in self.emotional_dimension:
                # Go through metrics, e.g., rmse, pcc, and ccc.
                for metric in self.metrics:
                    # Go through the sub-dictionary belonging to each subject.
                    main_record_dict[trial][emotion][metric].append(
                        epoch_record_dict[trial][emotion][metric]
                    )

        return main_record_dict