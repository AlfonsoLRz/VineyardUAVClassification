from imblearn.under_sampling import *
import numpy as np
import random

random_seed = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def stratified_sampling(samples, labels):
    """
    Performs stratified sampling.
    :param samples: Samples (X).
    :param labels: Labels.
    :return: Stratified samples.
    """
    sm = RandomUnderSampler(sampling_strategy='not minority', random_state=random_seed)
    x_balanced, y_balanced = sm.fit_resample(samples, labels)

    return x_balanced, y_balanced
