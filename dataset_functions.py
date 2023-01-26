import albumentations as A
from hypercube_set import *
from imblearn.over_sampling import *
from imblearn.under_sampling import *
from keras.utils import to_categorical
from randomness import *
from sklearn.model_selection import train_test_split


def augment_chunks(patch, label):
    """
    Augment patches by adding variations and random changes.
    """
    np.random.seed(random_seed)

    augmented_chunks = []
    augmented_chunk_labels = []

    for i in range(0, len(patch)):
        new_images = augment_chunk(patch[i])
        augmented_chunks.extend(new_images)
        augmented_chunk_labels.extend([label[i] for _ in new_images])

    return shuffle(np.asarray(augmented_chunks), np.asarray(augmented_chunk_labels))


def augment_chunk(img):
    """
    Augment a single patch by rotating it.
    """
    return img, np.fliplr(img), np.flipud(img), np.rot90(img, k=1, axes=(0, 1)), np.rot90(img, k=3, axes=(0, 1))


def balance_classes(patch, label, smote=True, clustering=True, reduce=False):
    """
    Balance the classes either by downsampling or upsampling.
    """
    if reduce:
        if clustering:
            sm = ClusterCentroids(sampling_strategy='not minority', random_state=random_seed)
        else:
            sm = RandomUnderSampler(sampling_strategy='not minority', random_state=random_seed)
    else:
        if smote:
            sm = SMOTE(sampling_strategy='not majority', random_state=random_seed)
        else:
            sm = RandomOverSampler(sampling_strategy='not majority', random_state=random_seed)

    sample_shape = patch[0].shape
    shape_length = 1
    for i in range(0, len(sample_shape)):
        shape_length *= sample_shape[i]

    reshaped_chunk = np.reshape(patch, (len(patch), shape_length))
    x_balanced, y_balanced = sm.fit_resample(reshaped_chunk, label)
    reshaped_chunk = np.reshape(x_balanced, (len(x_balanced),) + sample_shape)

    return shuffle(reshaped_chunk, y_balanced), np.delete(patch, sm.sample_indices_, axis=0), \
           np.delete(label, sm.sample_indices_, axis=0)


def get_center(data):
    """
    Get the centered pixel of train and test data.
    """
    shape = data[0].shape
    center = np.array([shape[0] // 2, shape[1] // 2])

    return data[:, center[0], center[1], :]


def hot_encode(label):
    """
    Hot encode the labels.
    """
    return to_categorical(label)


def reduce_labels_center(label):
    """
    Reduce the labels to the center of the patch.
    """
    new_labels = np.zeros((label.shape[0],), np.int32)
    patch_shape = label[0].shape
    patch_center = np.array([patch_shape[0] // 2, patch_shape[1] // 2])

    for i in range(0, len(label)):
        new_labels[i] = label[i][patch_center[0], patch_center[1]]

    return new_labels


def remove_soil_labels(patch, label, soil_label=0):
    """
    Remove soil labels from the data.
    """
    return patch[label != soil_label], label[label != soil_label]


def shuffle(patch, label):
    """
    Shuffle the data.
    """
    perm = np.random.RandomState(seed=random_seed).permutation(len(patch))
    return patch[perm], label[perm]


def split_train_test(patch, labels, test_size=0.7):
    """
    Split data into training and testing sets.
    """
    return train_test_split(patch, labels, test_size=test_size, shuffle=True, random_state=random_seed)
