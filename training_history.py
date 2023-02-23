import matplotlib.pyplot as plt
import numpy as np
import paths
import pickle as pickle


class TrainingHistory:
    """
    Class to store the training history of a model.
    """
    def __init__(self, accuracy_name='accuracy'):
        """
        Initialize the training history.
        """
        self._history = None
        self._epoch_marks = []
        self._accuracy_name = accuracy_name
        self._training_time = 0
        self._num_samples = 0

    def append_history(self, history, training_callback=None, samples=None):
        """
        Append the history of a model to the training history.
        :param history: New training history.
        """
        if self._history is None:
            self._history = history
        else:
            for key in self._history.keys():
                self._history[key] = np.append(self._history[key], history[key])

        self._epoch_marks.append(len(history['val_loss']))

        if training_callback is not None:
            self._training_time += training_callback.duration

        if samples is not None:
            self._num_samples += samples.shape[0]

        print("Samples: " + str(self._num_samples) + ", Time: " + str(self._training_time))

    def get_accuracy_key(self):
        """
        Get the key of the accuracy metric.
        :return: Accuracy metric key.
        """
        return self._accuracy_name

    def get_history(self):
        """
        Get the training history.
        :return: Training history.
        """
        return self._history

    def get_history_length(self):
        """
        Get the length of the training history.
        :return: Length of training history.
        """
        return len(self._history['val_loss'])

    def get_training_time(self):
        """
        Get the training time.
        :return: Training time.
        """
        return self._training_time

    @staticmethod
    def load(model_name):
        """
        Load the training history from file.
        :param model_name: Name of model.
        """
        return pickle.load(open(paths.result_folder + 'history/' + model_name + '.p', 'rb'))

    def save(self, model_name):
        """
        Save the history object to file.
        :param model_name: Name of model.
        """
        pickle.dump(self, open(paths.result_folder + 'history/' + model_name + '.p', 'wb'))
