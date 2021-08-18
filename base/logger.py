import os

import statistics
import numpy as np

from scipy.stats import pearsonr


class PlotHandlerTrial(object):
    r"""
    A class to plot the output-label figures.
    """

    def __init__(self, metrics, emotional_dimension, epoch_result_dict,
                 trialwise_output_dict, trialwise_continuous_label_dict,
                 epoch=None, train_mode=None, directory_to_save_plot=None):
        self.metrics = metrics
        self.emotional_dimension = emotional_dimension
        self.epoch_result_dict = epoch_result_dict

        self.epoch = epoch
        self.train_mode = train_mode
        self.directory_to_save_plot = directory_to_save_plot

        self.trialwise_output_dict = trialwise_output_dict
        self.trialwise_continuous_label_dict = trialwise_continuous_label_dict

    def complete_directory_to_save_plot(self):
        r"""
        Determine the full path to save the plot.
        """
        if self.train_mode:
            exp_folder = "train"
        else:
            exp_folder = "validate"

        if self.epoch is None:
            exp_folder = "test"

        directory = os.path.join(self.directory_to_save_plot, "plot", exp_folder, "epoch_" + str(self.epoch))
        if self.epoch == "test":
            directory = os.path.join(self.directory_to_save_plot, "plot", exp_folder)

        os.makedirs(directory, exist_ok=True)
        return directory

    def save_output_vs_continuous_label_plot(self):
        r"""
        Plot the output versus continuous label figures for each session.
        """

        for (trial, output_record), (_, label_record) in zip(self.trialwise_output_dict.items(), self.trialwise_continuous_label_dict.items()):

            complete_directory = self.complete_directory_to_save_plot()

            plot_filename = trial
            full_plot_filename = os.path.join(complete_directory, plot_filename + ".jpg")

            # Find the y ranges for subplot with better clarity.
            if len(self.emotional_dimension) > 1:
                ylim_low, ylim_high = [], []
                for emotion in self.emotional_dimension:
                    ylim_low.append(min(min(output_record[emotion]), min(label_record[emotion])))
                    ylim_high.append(max(max(output_record[emotion]), max(label_record[emotion])))
                ylim_low, ylim_high = min(ylim_low) * 1.15, max(ylim_high) * 1.15
            else:
                ylim_low, ylim_high = None, None

            self.plot_and_save(full_plot_filename, trial, output_record, label_record, ylim_low, ylim_high)

    def plot_and_save(self, full_plot_filename, trial, output, continuous_label, ylim_low=None, ylim_high=None):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(len(self.emotional_dimension), 1)

        for index, emotion in enumerate(self.emotional_dimension):
            result_list = []

            for metric in self.metrics:
                result = self.epoch_result_dict[trial][emotion][metric][0]
                # The pcc usually have two output, one for value and one for confidence. So
                # here we only read and the value and discard the confidence.
                if metric == "pcc":
                    result = self.epoch_result_dict[trial][emotion][metric][0][0]
                result_list.append(result)

            if len(self.emotional_dimension) > 1:
                # Plot the sub-figures, each for one emotional dimension.
                ax[index].plot(output[emotion], "r-", label="Output")
                ax[index].plot(continuous_label[emotion], "g-", label="Label")
                ax[index].set_ylim([ylim_low, ylim_high])
                ax[index].set_xlabel("Sample")
                ax[index].set_ylabel("Value")
                ax[index].legend(loc="upper right", framealpha=0.2)
                ax[index].title.set_text(
                    "{}: rmse={:.3f}, pcc={:.3f}, ccc={:.3f}".format(emotion, *result_list))
            else:
                ax.plot(output[emotion], "r-", label="Output")
                ax.plot(continuous_label[emotion], "g-", label="Label")
                ax.set_xlabel("Sample")
                ax.set_ylabel("Value")
                ax.legend(loc="upper right", framealpha=0.2)
                ax.title.set_text(
                    "{}: rmse={:.3f}, pcc={:.3f}, ccc={:.3f}".format(emotion, *result_list))
        fig.tight_layout()
        plt.savefig(full_plot_filename)
        plt.close()


class ConcordanceCorrelationCoefficient(object):
    """
    A class for performing concordance correlation coefficient (CCC) centering. Basically, when multiple continuous labels
    are available, it is not a good choice to perform a direct average. Formally, a Lin's CCC centering has to be done.

    This class is a Pythonic equivalence of CCC centering to the Matlab scripts ("run_gold_standard.m")
        from the AVEC2016 dataset.

    Ref:
        "Lawrence I-Kuei Lin (March 1989).  A concordance correlation coefficient to evaluate reproducibility".
            Biometrics. 45 (1): 255–268. doi:10.2307/2532051. JSTOR 2532051. PMID 2720055.
    """

    def __init__(self, data):
        self.data = data
        if data.shape[0] > data.shape[1]:
            self.data = data.T
        self.rator_number = self.data.shape[0]
        self.combination_list = self.generate_combination_pair()
        self.cnk_matrix = self.generate_cnk_matrix()
        self.ccc = self.calculate_paired_ccc()
        self.agreement = self.calculate_rator_wise_agreement()
        self.mean_data = self.calculate_mean_data()
        self.weight = self.calculate_weight()
        self.centered_data = self.perform_centering()

    def perform_centering(self):
        """
        The centering is done by directly average the shifted and weighted data.
        :return: (ndarray), the centered  data.
        """
        centered_data = self.data - np.repeat(self.mean_data[:, np.newaxis], self.data.shape[1], axis=1) + self.weight
        return centered_data

    def calculate_weight(self):
        """
        The weight of the m continuous labels. It will be used to weight (actually translate) the data when
            performing the final step.
        :return: (float), the weight of the given m continuous labels.
        """
        weight = np.sum((self.mean_data * self.agreement) / np.sum(self.agreement))
        return weight

    def calculate_mean_data(self):
        """
        A directly average of data.
        :return: (ndarray), the averaged data.
        """
        mean_data = np.mean(self.data, axis=1)
        return mean_data

    def generate_combination_pair(self):
        """
        Generate all possible combinations of Cn2.
        :return: (ndarray), the combination list of Cn2.
        """
        n = self.rator_number
        combination_list = []

        for boy in range(n - 1):
            for girl in np.arange(boy + 1, n, 1):
                combination_list.append([boy, girl])

        return np.asarray(combination_list)

    def generate_cnk_matrix(self):
        """
        Generate the Cn2 matrix. The j-th column of the matrix records all the possible candidate
            to the j-th rater. So that for the j-th column, we can acquire all the possible unrepeated
            combination for the j-th rater.
        :return:
        """
        total = self.rator_number
        cnk_matrix = np.zeros((total - 1, total))

        for column in range(total):
            cnk_matrix[:, column] = np.concatenate((np.where(self.combination_list[:, 0] == column)[0],
                                                    np.where(self.combination_list[:, 1] == column)[0]))

        return cnk_matrix.astype(int)

    @staticmethod
    def calculate_ccc(x, y):
        """
        Calculate the CCC.
        :param x: (ndarray), an 1xn array.
        :param y: (ndarray), another 1xn array.
        :return: the CCC.
        """

        x_mean = np.nanmean(x)
        y_mean = np.nanmean(y)
        covariance = np.nanmean((x - x_mean) * (y - y_mean))
        # Make it consistent with Matlab's nanvar (division by len(x)-1, not len(x)))
        x_var = 1.0 / (len(x) - 1) * np.nansum((x - x_mean) ** 2)
        y_var = 1.0 / (len(y) - 1) * np.nansum((y - y_mean) ** 2)

        concordance_correlation_coefficient = \
            (2 * covariance) / (x_var + y_var + (x_mean - y_mean) ** 2 + 1e-100)

        return concordance_correlation_coefficient

    def calculate_paired_ccc(self):
        """
        Calculate the CCC for all the pairs from the combination list.
        :return: (ndarray), the CCC for each combinations.
        """
        ccc = np.zeros((self.combination_list.shape[0]))
        for index in range(len(self.combination_list)):
            ccc[index] = self.calculate_ccc(self.data[self.combination_list[index, 0], :],
                                            self.data[self.combination_list[index, 1], :])

        return ccc

    def calculate_rator_wise_agreement(self):
        """
        Calculate the inter-rater CCC agreement.
        :return: (ndarray), a array recording the CCC agreement of each single rater to all the rest raters.
        """

        ccc_agreement = np.zeros(self.rator_number)

        for index in range(self.rator_number):
            ccc_agreement[index] = np.mean(self.ccc[self.cnk_matrix[:, index]])

        return ccc_agreement


class ContinuousOutputHandlerNPYTrial(object):
    def __init__(self, emotion_dimension):
        self.emotion_dimension = emotion_dimension
        self.seen_trials = []
        self.trialwise_records = {}
        self.partition_records = {emotion: [] for emotion in self.emotion_dimension}

    def update_output_for_seen_trials(self, output, trials, indices, lengths):

        for i, trial in enumerate(trials):

            # If this is the first time to record the output for trial
            if trial not in self.seen_trials:
                self.seen_trials.append(trial)
                self.trialwise_records[trial] = self.create_list_for_a_trial(lengths[i])

            index = indices[i]
            for j, emotion in enumerate(self.emotion_dimension):
                for k, data in enumerate(output[i, :, j]):
                    if k == lengths[i]:
                        break
                    self.trialwise_records[trial][emotion][index[k]].append(output[i, k, j])

    def average_trial_wise_records(self):

        for trial in self.seen_trials:
            for emotion in self.emotion_dimension:
                length = len(self.trialwise_records[trial][emotion])

                for i in range(length):
                    self.trialwise_records[trial][emotion][i] = statistics.mean(
                        self.trialwise_records[trial][emotion][i])

                self.trialwise_records[trial][emotion] = np.asarray(self.trialwise_records[trial][emotion])

    def concat_records(self):
        for trial in self.seen_trials:
            for emotion in self.emotion_dimension:
                self.partition_records[emotion].extend(self.trialwise_records[trial][emotion])

        for emotion in self.emotion_dimension:
            self.partition_records[emotion] = np.asarray(self.partition_records[emotion])

    def create_list_for_a_trial(self, length):
        trial_record = {emotion: [[] for i in range(length)] for emotion in self.emotion_dimension}
        return trial_record


class ContinuousMetricsCalculatorTrial(object):
    r"""
        A class to calculate the metrics, usually rmse, pcc, and ccc for continuous regression.
        """

    def __init__(
            self,
            metrics,
            emotional_dimension,
            output_handler,
            continuous_label_handler,
    ):

        # What metrics to calculate.
        self.metrics = metrics

        # What emotional dimensions to consider.
        self.emotional_dimension = emotional_dimension

        # The instances saving the data for evaluation.
        self.output_handler = output_handler
        self.continuous_label_handler = continuous_label_handler

        # Initialize the dictionary for saving the metric results.
        self.metric_record_dict = self.init_metric_record_dict()

    def get_partitionwise_output_and_continuous_label(self):
        return self.output_handler.partition_records, \
               self.continuous_label_handler.partition_records

    def get_trialwise_output_and_continuous_label(self):
        return self.output_handler.trialwise_records, \
               self.continuous_label_handler.trialwise_records

    def init_metric_record_dict(self):
        trialwise_dict, _ = self.get_trialwise_output_and_continuous_label()
        metric_record_dict = {key: [] for key in trialwise_dict}
        return metric_record_dict

    @staticmethod
    def calculator(output, label, metric):
        if metric == "rmse":
            result = np.sqrt(((output - label) ** 2).mean())
        elif metric == "pcc":
            result = pearsonr(output, label)
        elif metric == "ccc":
            result = ConcordanceCorrelationCoefficient.calculate_ccc(output, label)
        else:
            raise ValueError("Metric {} is not defined.".format(metric))
        return result

    def calculate_metrics(self):

        # Load the data for three scenarios.
        # They will all be evaluated.
        trialwise_output, trialwise_continuous_label = self.get_trialwise_output_and_continuous_label()
        partitionwise_output, partitionwise_continuous_label = self.get_partitionwise_output_and_continuous_label()

        for (trial_id, output), (_, label) in zip(
                trialwise_output.items(), trialwise_continuous_label.items()):

            trial_record_dict = {key: {} for key in self.emotional_dimension}

            for column, emotion in enumerate(self.emotional_dimension):
                result_dict = {metric: [] for metric in self.metrics}
                for metric in self.metrics:
                    result = self.calculator(output[emotion], label[emotion], metric)
                    result_dict[metric] = [result]
                    # if metric == "pcc":
                    #     result_dict[metric] = [result]
                trial_record_dict[emotion] = result_dict
            self.metric_record_dict[trial_id] = trial_record_dict

        # Partition-wise evaluation
        overall_records = {emotion: [] for emotion in self.emotional_dimension}
        for emotion in self.emotional_dimension:
            partitionwise_dict = {metric: [] for metric in self.metrics}

            for metric in self.metrics:
                result = self.calculator(partitionwise_output[emotion], partitionwise_continuous_label[emotion],
                                         metric)

                if metric == "pcc":
                    partitionwise_dict[metric].append(result)
                else:
                    partitionwise_dict[metric].append([result])

            overall_records[emotion] = partitionwise_dict

        self.metric_record_dict['overall'] = overall_records
