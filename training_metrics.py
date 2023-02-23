import numpy as np
import pickle
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, roc_auc_score

import paths


class TrainingMetrics:
    def __init__(self):
        self._test_ao = []
        self._test_aa = []
        self._test_kappa = []
        self._test_f1 = []
        self._test_roc_auc = []

    def append_evaluation(self, y_true, y_pred, y_pred_prob):
        self._test_ao.append(self.__get_overall_accuracy(y_true, y_pred))
        self._test_aa.append(self.__get_average_accuracy(y_true, y_pred))
        self._test_kappa.append(self.__get_kappa_loss(y_pred, y_true))
        self._test_f1.append(self.__get_f1_score(y_true, y_pred))
        self._test_roc_auc.append(self.__get_roc_auc_score(y_true, y_pred_prob))

    def get_oa(self):
        return self._test_ao

    def get_aa(self):
        return self._test_aa

    def get_kappa(self):
        return self._test_kappa

    def get_f1(self):
        return self._test_f1

    def get_roc_auc(self):
        return self._test_roc_auc

    def get_oa_variance(self):
        return np.std(self._test_ao)

    def get_aa_variance(self):
        return np.std(self._test_aa)

    def get_kappa_variance(self):
        return np.std(self._test_kappa)

    def get_f1_variance(self):
        return np.std(self._test_f1)

    def get_roc_auc_variance(self):
        return np.std(self._test_roc_auc)

    @staticmethod
    def load(network_name):
        return pickle.load(open(paths.result_folder + 'metrics/' + network_name + '.p', 'rb'))

    def print_metrics(self):
        print("Overall accuracy: " + str(np.mean(self._test_ao)) + " +- " + str(np.std(self._test_ao)))
        print("Average accuracy: " + str(np.mean(self._test_aa)) + " +- " + str(np.std(self._test_aa)))
        print("Kappa loss: " + str(np.mean(self._test_kappa)) + " +- " + str(np.std(self._test_kappa)))
        print("F1 score: " + str(np.mean(self._test_f1)) + " +- " + str(np.std(self._test_f1)))
        print("ROC AUC score: " + str(np.mean(self._test_roc_auc)) + " +- " + str(np.std(self._test_roc_auc)))

    def save(self, network_name):
        with open(paths.result_folder + 'metrics/' + network_name + '.p', 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def __get_overall_accuracy(y_true, y_pred):
        return accuracy_score(y_true, y_pred, normalize=True)

    @staticmethod
    def __get_average_accuracy(y_true, y_pred):
        # Get distinct values in y_true
        distinct_values = np.unique(y_true)

        # Get accuracy for each class
        accuracies = []
        for value in distinct_values:
            # Get indices for current class
            indices = np.where(y_true == value)

            # Get accuracy for current class
            accuracies.append(accuracy_score(y_true[indices], y_pred[indices]))

        # Return average accuracy
        return np.mean(accuracies)

    @staticmethod
    def __get_kappa_loss(y_pred, y_true):
        return cohen_kappa_score(y_true, y_pred, weights='quadratic')

    @staticmethod
    def __get_f1_score(y_true, y_pred):
        return f1_score(y_true, y_pred, average='weighted')

    @staticmethod
    def __get_roc_auc_score(y_true, y_pred):
        return roc_auc_score(y_true, y_pred, average='weighted', multi_class='ovr')
