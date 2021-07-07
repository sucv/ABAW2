import os
import time
import copy
from tqdm import tqdm
from itertools import cycle


import numpy as np
import torch
from torch import optim
from torch.nn import MSELoss
import torch.utils.data
from sklearn.metrics import f1_score

from base.logger import ContinuousOutputHandlerNPYTrial, ContinuousMetricsCalculatorTrial, ConcordanceCorrelationCoefficient, PlotHandlerTrial


def calculate_confusion_matrix(num_classes, preds, labels):
    confusion_matrix = np.zeros((num_classes + 1, num_classes + 1))
    for label, pred in zip(labels, preds):
        confusion_matrix[label, pred] += 1

    confusion_matrix[-1, :] = np.sum(confusion_matrix, axis=0)
    confusion_matrix[:, -1] = np.sum(confusion_matrix, axis=1)

    confusion_matrix[:-1, :-1] /= confusion_matrix[:-1, -1][:, np.newaxis]
    confusion_matrix[:-1, :-1] = np.around(confusion_matrix[:-1, :-1], decimals=3)
    return confusion_matrix.tolist()


def averaged_f1_score(input, target):
    N, label_size = input.shape
    average_type = 'binary' if target.max() <= 1 else None
    f1s = []
    for i in range(label_size):
        f1 = f1_score(input[:, i], target[:, i], average=average_type)
        f1s.append(f1.tolist())
    return np.mean(f1s), f1s


def accuracy(input, target):
    assert len(input.shape) == 1
    return sum(input==target)/input.shape[0]


def averaged_accuracy(x, y):
    assert len(x.shape) == 2
    N, C =x.shape
    accs = []
    for i in range(C):
        acc = accuracy(x[:, i], y[:, i])
        accs.append(acc)
    return np.mean(accs), accs, calculate_confusion_matrix(7, x, y) if C == 1 else None


class ABAW2Trainer(object):
    def __init__(self, model, model_name='2d1d', save_path=None, train_emotion='both', head='multi-headed', factor=0.1,
                 early_stopping=100, criterion=None, milestone=[0], patience=10, learning_rate=0.00001, device='cpu', num_classes=2, max_epoch=50, min_learning_rate=1e-7,
                 emotional_dimension=None, metrics=None, verbose=False, print_training_metric=False, save_plot=False, window_length=300,
                 load_best_at_each_epoch=False, fold=0, au_n=12, pred_AU=0, pred_Expre=0, **kwargs):

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
        self.resume = False
        self.time_fit_start = None
        self.combined_record_dict = {'train': {}, 'validate': {}}
        self.train_losses = []
        self.validate_losses = []
        self.csv_filename = None
        self.best_epoch_info = None
        self.load_best_at_each_epoch = load_best_at_each_epoch
        self.au_n = au_n
        self.pred_AU = pred_AU
        self.pred_Expre = pred_Expre

    def init_optimizer_and_scheduler(self):
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

    def train(self, data_loader, epoch, adversarial):
        self.model.train()
        loss, result_dict = self.loop(data_loader, epoch,
                                      train_mode=True, adversarial=adversarial)
        return loss, result_dict

    def validate(self, data_loader, epoch):
        with torch.no_grad():
            self.model.eval()
            loss, result_dict = self.loop(data_loader, epoch, train_mode=False)
        return loss, result_dict

    def fit(
            self,
            data_to_load,
            min_num_epochs=10,
            adversarial=0,
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
                'ccc': -1e10,
                'classification_mean': -1e10
            }

        # Loop the epochs
        for epoch in np.arange(start_epoch, self.max_epoch):

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

            if adversarial:
                data_to_load_for_adversarial = {'train': data_to_load['train'], 'target': data_to_load['target']}
                train_loss_dict, train_record_dict = self.train(data_to_load_for_adversarial, epoch, adversarial)
            else:
                train_loss_dict, train_record_dict = self.train(data_to_load['train'], epoch, adversarial)

            # Combine the record to a long array for each subject.
            # self.combined_record_dict['train'] = self.combine_record_dict(
            #     self.combined_record_dict['train'], train_record_dict)

            validate_loss_dict, validate_record_dict = self.validate(data_to_load['validate'], epoch)

            # self.combined_record_dict['validate'] = self.combine_record_dict(
            #     self.combined_record_dict['validate'], validate_record_dict)

            self.train_losses.append(train_loss_dict['epoch_loss'])
            self.validate_losses.append(validate_loss_dict['epoch_loss'])

            improvement = False
            if self.head == "no-headed":
                validate_ccc = 0
            else:
                if self.train_emotion == "both":
                    validate_ccc = np.mean(
                        [validate_record_dict['overall'][emotion]['ccc'] for emotion in self.emotional_dimension])
                elif self.train_emotion == "arousal":
                    # validate_ccc = np.mean(validate_record_dict['overall']['Arousal']['ccc'])
                    validate_ccc = np.mean(
                        [validate_record_dict['overall'][emotion]['ccc'] for emotion in self.emotional_dimension])
                elif self.train_emotion == "valence":
                    # validate_ccc = np.mean(validate_record_dict['overall']['Valence']['ccc'])
                    validate_ccc = np.mean(
                        [validate_record_dict['overall'][emotion]['ccc'] for emotion in self.emotional_dimension])
                else:
                    raise ValueError("Unknown emotion dimension!")
            classification_totals = [one_acc_f1['total'] for one_acc_f1 in validate_record_dict['classification']]
            classification_total_mean = np.mean(classification_totals) if len(classification_totals) > 0 else 0
            # If a lower validate loss appears.
            if validate_ccc + classification_total_mean > self.best_epoch_info['ccc'] + self.best_epoch_info['classification_mean']:
                if save_model:
                    torch.save(self.model.state_dict(), os.path.join(self.save_path, "model_state_dict.pth"))

                improvement = True
                self.best_epoch_info = {
                    'model_weights': copy.deepcopy(self.model.state_dict()),
                    'loss': validate_loss_dict['epoch_loss'],
                    'ccc': validate_ccc,
                    'classification_mean': classification_total_mean,
                    'epoch': epoch,
                    'scalar_metrics': {
                        'train_loss': train_loss_dict['epoch_loss'],
                        'validate_loss': validate_loss_dict['epoch_loss'],
                    },
                    'array_metrics': {
                        'train_metric_record': train_record_dict['overall'],
                        'validate_metric_record': validate_record_dict['overall']
                    }
                }

            if validate_loss_dict['epoch_loss'] < 0:
                print('validate loss negative')

            if self.verbose:
                print(
                    "\n Fold {:2} Epoch {:2} in {:.0f}s || Train loss={:.3f} | Val loss={:.3f} | LR={:.1e} | Release_count={} | best={} | "
                    "improvement={}-{}".format(
                        self.fold,
                        epoch + 1,
                        time.time() - time_epoch_start,
                        train_loss_dict['epoch_loss'],
                        validate_loss_dict['epoch_loss'],
                        self.optimizer.param_groups[0]['lr'],
                        parameter_controller.release_count,
                        int(self.best_epoch_info['epoch']) + 1,
                        improvement,
                        self.early_stopping_counter))

                print(train_record_dict['overall'])
                print(train_record_dict['classification'])
                print(validate_record_dict['overall'])
                print(validate_record_dict['classification'])
                print("------")

            checkpoint_controller.save_log_to_csv(
                epoch, train_record_dict['overall'], validate_record_dict['overall'],
                cls_train_record=train_record_dict['classification'],
                cls_val_record=validate_record_dict['classification'])

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

    def loop(self, data_loader, epoch, train_mode=True, adversarial=0):
        running_loss = 0.0
        running_loss_source = 0.0
        running_loss_target = 0.0
        running_loss_AU = 0.0
        running_loss_Expre = 0.0
        AU_pred_collection = []
        AU_label_collection = []
        Expre_pred_collection = []
        Expre_label_collection = []

        output_handler = ContinuousOutputHandlerNPYTrial(self.emotional_dimension)
        continuous_label_handler = ContinuousOutputHandlerNPYTrial(self.emotional_dimension)

        # This object calculate the metrics, usually by root mean square error, pearson correlation
        # coefficient, and concordance correlation coefficient.
        metric_handler = ContinuousMetricsCalculatorTrial(self.metrics, self.emotional_dimension,
                                                          output_handler, continuous_label_handler)
        total_batch_counter = 0

        if train_mode and adversarial:
            target_loader = data_loader['target']
            data_loader = data_loader['train']
            target_loader_iter = cycle(iter(target_loader))
            criterion_domain = torch.nn.NLLLoss()
        if self.pred_AU:
            from base.FocalLoss import BCEFocalLoss
            criterion_AU = BCEFocalLoss()
        if self.pred_Expre:
            from base.FocalLoss import FocalLoss
            # criterion_Expre = FocalLoss()
            criterion_Expre = torch.nn.CrossEntropyLoss()

        len_dataloader = len(data_loader)

        for batch_index, (X, Y, trials, lengths, indices) in tqdm(enumerate(data_loader), total=len_dataloader):
            total_batch_counter += len(trials)

            p = float(batch_index + epoch * len_dataloader) / self.max_epoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            inputs = X['frame'].to(self.device)
            labels = Y.float().to(self.device)
            AUs = X['AUs']
            Expres = X['Expres']
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

            outputs = self.model(inputs, alpha=alpha)

            if self.head != "no-headed":
                output_handler.update_output_for_seen_trials(outputs['x'].detach().cpu().numpy(), trials, indices, lengths)
                continuous_label_handler.update_output_for_seen_trials(labels.detach().cpu().numpy(), trials, indices, lengths)
                loss = self.criterion(outputs['x'], labels, loss_weights)
            else:
                loss = 0

            if train_mode and adversarial:
                data_target = next(target_loader_iter)
                T, _, _, _, _ = data_target

                if 'frame' in T:
                    input_target = T['frame'].to(self.device)

                    # For the last batch, the batch size may be different between the source and target domain.
                    # This line ensures the size to be identical.
                    input_target = input_target[:len(inputs)]

                domain_label = torch.zeros(input_target.shape[0] * input_target.shape[1]).long().to(self.device)

                outputs_source = outputs['domain_output']
                loss_source = criterion_domain(outputs_source, domain_label)

                domain_label = torch.ones(input_target.shape[0] * input_target.shape[1]).long().to(self.device)

                outputs_target = self.model(input_target, alpha=alpha)
                outputs_target = outputs_target['domain_output']

                loss_target = criterion_domain(outputs_target, domain_label)

                loss = loss + loss_source + loss_target

                running_loss_source += loss_source.mean().item()
                running_loss_target += loss_target.mean().item()

            if self.pred_AU:
                AU_label, AU_mask = AUs
                if len(AU_label[AU_mask]) > 0:
                    AU_loss = criterion_AU(outputs['AU'][AU_mask], AU_label[AU_mask].to(self.device))
                else:
                    AU_loss = 0
                loss = loss + 0.1 * AU_loss
                if len(AU_label[AU_mask]) > 0:
                    AU_pred_collection.append(outputs['AU'][AU_mask].detach().cpu().reshape(-1, self.au_n).numpy())
                    AU_label_collection.append(AU_label[AU_mask].detach().cpu().reshape(-1, self.au_n).numpy())

            if self.pred_Expre:
                EX_label, EX_mask = Expres
                EX_label = EX_label.long()
                expres_n = outputs['Expre'].shape[-1]
                if len(EX_label[EX_mask]) > 0:
                    EX_loss = criterion_Expre(outputs['Expre'][EX_mask.repeat(1, 1, expres_n)].reshape(-1, expres_n),
                                              EX_label[EX_mask].to(self.device))
                else:
                    EX_loss = 0
                loss = loss + 0.1 * EX_loss
                if len(EX_label[EX_mask]) > 0:
                    outputs['Expre'] = torch.softmax(outputs['Expre'], dim=-1)
                    Expre_pred_collection.append(
                        outputs['Expre'][EX_mask.repeat(1, 1, expres_n)].detach().cpu().reshape(-1, expres_n).numpy()
                    )
                    Expre_label_collection.append(EX_label[EX_mask].detach().cpu().reshape(-1, 1).numpy())
            if type(loss) is float:
                continue
            running_loss += loss.mean().item()

            if train_mode:
                loss.backward()
                self.optimizer.step()

        epoch_loss_dict = {}

        epoch_loss = running_loss / total_batch_counter
        epoch_loss_dict['epoch_loss'] = epoch_loss
        if train_mode and adversarial:
            epoch_loss_dict['source'] = running_loss_source / total_batch_counter
            epoch_loss_dict['target'] = running_loss_target / total_batch_counter

        if self.pred_AU:
            epoch_loss_dict['AU'] = running_loss_AU / total_batch_counter

        if self.pred_Expre:
            epoch_loss_dict['Expre'] = running_loss_Expre / total_batch_counter

        output_handler.average_trial_wise_records()
        continuous_label_handler.average_trial_wise_records()
        output_handler.concat_records()
        continuous_label_handler.concat_records()

        # Compute the root mean square error, pearson correlation coefficient and significance, and the
        # concordance correlation coefficient.
        # They are calculated by  first concatenating all the output
        # and continuous labels to two long arrays, and then calculate the metrics.
        if self.head != "no-headed":
            metric_handler.calculate_metrics()
            epoch_result_dict = metric_handler.metric_record_dict
        else:
            epoch_result_dict = {'overall': 0}

        AU_metric, Expre_metric = [], []
        if self.pred_AU:
            AU_pred_collection = np.concatenate(AU_pred_collection).round()
            AU_label_collection = np.concatenate(AU_label_collection)
            acc = averaged_accuracy(AU_label_collection, AU_pred_collection)
            f1 = averaged_f1_score(AU_label_collection, AU_pred_collection)
            AU_metric = [{'acc': acc, 'f1': f1, 'total': 0.5*f1[0]+0.5*acc[0]}]
        if self.pred_Expre:
            Expre_pred_collection = np.concatenate(Expre_pred_collection)
            Expre_label_collection = np.concatenate(Expre_label_collection)
            Expre_pred_collection = np.stack([np.array([p.argmax()]) for p in Expre_pred_collection])
            acc = averaged_accuracy(Expre_label_collection, Expre_pred_collection)
            f1 = averaged_f1_score(Expre_label_collection, Expre_pred_collection)
            Expre_metric = [{'acc': acc, 'f1': f1, 'total': 0.67*f1[0]+0.33*acc[0]}]

        epoch_result_dict['classification'] = AU_metric + Expre_metric
        if self.save_plot:
            # This object plot the figures and save them.
            plot_handler = PlotHandlerTrial(self.metrics, self.emotional_dimension, epoch_result_dict,
                                            output_handler.trialwise_records,
                                            continuous_label_handler.trialwise_records,
                                            epoch=epoch, train_mode=train_mode,
                                            directory_to_save_plot=self.save_path)
            plot_handler.save_output_vs_continuous_label_plot()

        return epoch_loss_dict, epoch_result_dict

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
        # pass
