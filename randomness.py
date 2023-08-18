from imblearn.under_sampling import *
import numpy as np
import random

random_seed = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def stratified_sampling(samples, labels, use_float=True, num_reduced_classes=2):
    """
    Performs stratified sampling.
    :param samples: Samples (X).
    :param labels: Labels.
    :return: Stratified samples.
    """
    # Flatten samples
    flatten_samples = samples
    if len(samples.shape) > 2:
        flatten_samples = samples.reshape(samples.shape[0], -1)

    # Count the number of samples per class
    if use_float:
        unique, counts = np.unique(labels, return_counts=True)
        ordered_classes = np.unravel_index(np.argsort(counts.ravel()), counts.shape)[0]

        for i in range(1, num_reduced_classes + 1):
            ref_count = counts[ordered_classes[-i - 1]]

            for j in range(1, i + 1):
                counts[ordered_classes[-j]] = ref_count
            # first_largest_class = ordered_classes[-1]
            # second_largest_class = ordered_classes[-2]
            # counts[first_largest_class] = counts[second_largest_class]

        # Build dict
        sampling_strategy = dict(zip(unique, counts))
        sm = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_seed)
    else:
        sm = RandomUnderSampler(sampling_strategy='not minority', random_state=random_seed)

    x_balanced, y_balanced = sm.fit_resample(flatten_samples, labels)
    x_balanced = x_balanced.reshape(x_balanced.shape[0], samples.shape[1], samples.shape[2], samples.shape[3])

    return x_balanced, y_balanced
