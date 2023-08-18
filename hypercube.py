import cv2
from enum import Enum
import math
import numpy as np


class VegetationIndex(Enum):
    NDVI = 0
    GDVI = 1
    MAX_MIN = 2             # Custom for visualization purposes


class Hypercube:
    RED_WL = 670
    GREEN_WL = 540
    BLUE_WL = 480
    NIR_WL = 800

    def __init__(self, hc_numpy, hc_mask, bands, path=None, baseline_class_idx=0, color_dict=None,
                 null_class_idx=0, null_idx=0):
        """
        Initializes a hypercube.
        :param hc_numpy: Hypercube.
        :param hc_mask: Class mask.
        :param path: System path to the hypercube.
        """
        self._hypercube = hc_numpy
        self._mask = hc_mask
        self._bands = bands
        self._path = path
        self._removed_indices = None
        self._train_indices = None
        self._test_indices = None
        self._color_dict = color_dict

        print("Hypercube shape: {}".format(self._hypercube.shape))

        self._to_id_image(baseline_class_idx, null_idx=null_idx, null_class=null_class_idx)
        self._labels = np.unique(self._mask)

    def calculate_index(self, thresholding=True, threshold=.5, index=VegetationIndex.NDVI):
        """
        Calculates a vegetation index.
        """
        nir_band = np.array(self._hypercube[:, :, self.__search_nearest_layer(self._bands, self.NIR_WL)])
        green_band = np.array(self._hypercube[:, :, self.__search_nearest_layer(self._bands, self.GREEN_WL)])
        red_band = np.array(self._hypercube[:, :, self.__search_nearest_layer(self._bands, self.RED_WL)])

        if index == VegetationIndex.NDVI:
            index_factor = (nir_band - red_band) / (nir_band + red_band)
        elif index == VegetationIndex.MAX_MIN:
            last_band = self._hypercube[:, :, self.get_shape()[2] // 2]
            first_band = self._hypercube[:, :, 0]
            index_factor = last_band - first_band
        else:
            index_factor = (nir_band - green_band) / (nir_band + green_band)

        if thresholding:
            index_factor = np.where(index_factor > threshold, 255.0, .0)

        return index_factor

    def flatten_hc(self):
        shape = self._hypercube.shape
        return np.reshape(self._hypercube,
                         (self._hypercube.shape[0] * self._hypercube.shape[1], self._hypercube.shape[2])), shape

    def flatten_mask(self):
        shape = self._mask.shape
        return np.reshape(self._mask, (self._mask.shape[0] * self._mask.shape[1])), shape

    def filter_wl(self, min_nm=0, max_nm=2000):
        """
        Retrieves a subset of hypercube layers.
        :param min_nm: Minimum nm to be selected.
        :param max_nm: Maximum nm to be selected.
        """
        min_idx, max_idx = 0, 0
        for i in range(len(self._bands)):
            if self._bands[i] < min_nm:
                min_idx = i
            if self._bands[i] < max_nm:
                max_idx = i

        self._hypercube = self._hypercube[:, :, min_idx + 1:max_idx]
        self._bands = self._bands[min_idx + 1:max_idx]

    def fit_model(self, standardizer):
        """
        Fits a standardizer to the hypercube.
        :param standardizer: Standardizer.
        """
        self._hypercube, shape = self.flatten_hc()
        standardizer.fit(self._hypercube[self._train_indices, :])
        self._hypercube = np.reshape(self._hypercube, shape)

    def get_bands(self):
        """
        :return: Hypercube bands.
        """
        return self._bands

    def get_class_mask(self):
        """
        :return: Class mask.
        """
        return self._mask

    def get_color_dict(self):
        """
        :return: Color dictionary.
        """
        return self._color_dict

    def get_hypercube(self):
        """
        :return: Hypercube.
        """
        return self._hypercube

    def get_labels(self):
        """
        :return: Labels.
        """
        return self._labels

    def get_num_samples(self, label):
        """
        Retrieves the number of samples for a specific label.
        :param label: Label.
        """
        return np.count_nonzero(self._mask == label)

    def get_patches(self, patch_size, patch_overlap, start_percentage, end_percentage, train=True):
        indices = self._train_indices if train else self._test_indices
        start_idx = math.floor(len(indices) * start_percentage)
        end_idx = math.floor(len(indices) * end_percentage)
        indices = indices[start_idx:end_idx]
        patches = []
        labels = []

        pre_patch_size = patch_size // 2
        post_patch_size = patch_size // 2 + 1 if patch_size % 2 != 0 else patch_size // 2
        patch_overlap = patch_size - patch_overlap

        for i in indices:
            x = i % self._mask.shape[1]
            y = i // self._mask.shape[1]
            if x + post_patch_size >= self._mask.shape[1] or y + post_patch_size >= self._mask.shape[0]\
                    or x - pre_patch_size < 0 or y - pre_patch_size < 0 or x % patch_overlap != 0 or y % patch_overlap != 0:
                continue

            patch = self._hypercube[y - pre_patch_size:y + post_patch_size, x - pre_patch_size:x + post_patch_size, :]
            patches.append(patch)
            labels.append(self._mask[y, x])

        return np.asarray(patches), np.asarray(labels) - 1

    @staticmethod
    def get_rgb_indices(bands):
        """
        Retrieves RGB indices from a hypercube.
        :param hc: Hypercube.
        :return: RGB indices.
        """
        red = Hypercube.__search_nearest_layer(bands, Hypercube.RED_WL)
        green = Hypercube.__search_nearest_layer(bands, Hypercube.GREEN_WL)
        blue = Hypercube.__search_nearest_layer(bands, Hypercube.BLUE_WL)

        return red, green, blue

    def get_shape(self):
        """
        :return: Hypercube shape.
        """
        return self._hypercube.shape

    def print_metadata(self, title=''):
        """
        Prints hypercube metadata.
        """
        print(title + 'Min: {}, Max: {}, Size: {}'.format(np.min(self._hypercube), np.max(self._hypercube),
                                                          self._hypercube.shape))

    def remove_ground_indices(self, ground_index):
        """
        Removes ground indices from hypercube.
        :param ground_index: Ground index.
        """
        # Flatten mask
        self._mask = np.reshape(self._mask, (self._mask.shape[0] * self._mask.shape[1]))
        self._removed_indices = np.where(self._mask == ground_index)[0]

        # Reshape again
        self._mask = np.reshape(self._mask, (self._hypercube.shape[0], self._hypercube.shape[1]))

    @staticmethod
    def save_mask(mask, path):
        """
        Saves class mask as .png file.
        :param mask: Class mask.
        :param path: Path to save the mask.
        """
        cv2.imwrite(path, mask)

    def split(self, train_percentage):
        """
        Splits the hypercube into training and test sets.
        :param train_percentage: Training percentage.
        :return: Training and test sets.
        """
        self._mask, mask_shape = self.flatten_mask()
        trainable_indices = np.arange(self._mask.shape[0])
        self._mask = np.reshape(self._mask, mask_shape)
        trainable_indices = np.delete(trainable_indices, self._removed_indices)
        train_size = math.floor(train_percentage * len(trainable_indices))
        self._train_indices = np.random.choice(trainable_indices, train_size, replace=False)
        self._test_indices = np.setdiff1d(trainable_indices, self._train_indices)
        # Shuffle
        self._test_indices = np.random.permutation(self._test_indices)

        print('Train size: {}, Test size: {}'.format(len(self._train_indices), len(self._test_indices)))

        del self._removed_indices
        del trainable_indices
        self._removed_indices = None

    def transform(self, model, num_features=-1):
        """
        Standardizes the hypercube.
        :param model: Standardizer, Feature reduction/transformation.
        """
        if num_features == -1:
            num_features = self._hypercube.shape[-1]

        self._hypercube, shape = self.flatten_hc()
        self._hypercube = model.transform(self._hypercube)
        self._hypercube = np.reshape(self._hypercube, (shape[0], shape[1], num_features))

    def subsample(self, subsampling_percentage):
        """
        Subsamples the hypercube.
        :param subsampling_percentage: Subsampling percentage.
        """
        starting_shape = self._hypercube.shape
        random_indices = np.random.choice(self._hypercube.shape[0] * self._hypercube.shape[1],
                                            math.floor(self._hypercube.shape[0] * self._hypercube.shape[1] *
                                                         subsampling_percentage), replace=False)
        self._hypercube = np.reshape(self._hypercube, (self._hypercube.shape[0] * self._hypercube.shape[1],
                                                       self._hypercube.shape[2]))
        self._mask = np.reshape(self._mask, (self._mask.shape[0] * self._mask.shape[1],))

        subsamples = self._hypercube[random_indices, :].copy()
        labels = self._mask[random_indices].copy()

        self._hypercube = np.reshape(self._hypercube, starting_shape)
        self._mask = np.reshape(self._mask, (starting_shape[0], starting_shape[1]))

        return subsamples, labels


    @staticmethod
    def __search_nearest_layer(bands, wl):
        """
        Searches for the nearest layer to a given wavelength.
        :param bands: Hypercube bands.
        :param wl: Wavelength float to be sought.
        :return: Nearest layer index.
        """
        begin, end, index = 0, len(bands) - 1, 0

        while True:
            middle = math.floor((begin + end) / 2.0)

            if begin == end or bands[middle] <= wl < bands[middle + 1]:
                index = middle
                break
            elif wl < bands[middle]:
                end = middle - 1
            else:
                begin = middle + 1

        if index == (len(bands) - 1):
            return index
        else:
            if abs(wl - bands[index]) < abs(wl - bands[index + 1]):
                return index
            else:
                return index + 1
    def _to_id_image(self, baseline_class_idx=0, null_idx=0, null_class=0):
        h = self._mask.shape[0]
        w = self._mask.shape[1]
        id_image = np.zeros(shape=(h, w))
        if self._color_dict is None:
            if len(self._mask.shape) == 3:
                self._color_dict = {(null_class, null_class, null_class): null_idx}
            else:
                self._color_dict = {null_class: null_idx}

        # Load color_dict with pickle
        # with open('color_dict_2022.pkl', 'rb') as handle:
        #     color_dict = pickle.load(handle)

        for y in range(0, h):
            for x in range(0, w):
                if len(self._mask.shape) == 3:
                    color = (int(self._mask[y, x, 0]), int(self._mask[y, x, 1]), int(self._mask[y, x, 2]))
                else:
                    color = (int(self._mask[y, x]))

                if color not in self._color_dict:
                    self._color_dict[color] = len(self._color_dict)

                if self._color_dict[color] == null_idx:
                    id_image[y, x] = null_idx
                else:
                    id_image[y, x] = self._color_dict[color] + baseline_class_idx

        # print(self._color_dict)
        # unique_ids = np.unique(id_image)
        # for i in range(0, len(unique_ids)):
        #     # Count number of pixels for each class
        #     num_pixels = np.count_nonzero(id_image == unique_ids[i])
        #     print("Class " + str(unique_ids[i]) + " has " + str(num_pixels) + " pixels.")

        # Save color_dict to file as an object
        # with open('color_dict_2022.pkl', 'wb') as f:
        #     pickle.dump(color_dict, f, pickle.HIGHEST_PROTOCOL)
        print(self._color_dict)
        self._mask = id_image
