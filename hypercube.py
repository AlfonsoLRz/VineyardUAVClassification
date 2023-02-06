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

    def __init__(self, hc_numpy, hc_mask, bands, path=None):
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

    def flatten(self):
        return np.reshape(self._hypercube,
                          (self._hypercube.shape[0] * self._hypercube.shape[1], self._hypercube.shape[2]))

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

    def get_hypercube(self):
        """
        :return: Hypercube.
        """
        return self._hypercube

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

    @staticmethod
    def save_mask(mask, path):
        """
        Saves class mask as .png file.
        :param mask: Class mask.
        :param path: Path to save the mask.
        """
        cv2.imwrite(path, mask)

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
