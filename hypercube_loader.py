import cv2
import glob

import rendering
from hypercube import Hypercube
import matplotlib.pyplot as plt
import numpy as np
import os
import paths
import scipy.io as sio
import spectral


def __load_hc(path, plot_hc=False):
    """
    Loads an hypercube from a given path.
    :param path: System path of mask.
    :param plot_hc: Renders an hypercube layer if enabled.
    :return: Hypercube.
    """
    hc_instance = spectral.open_image(path)
    hc = hc_instance.load()

    if hc is not None and plot_hc:
        plt.imshow(hc[:, :, hc.shape[2] // 2])
        plt.show()

    return hc, hc_instance.bands.centers


def __load_mask(path, plot_mask=False):
    """
    Loads a mask from a given path.
    :param path: System path of mask.
    :param plot_mask: Renders mask if enabled.
    :return: Mask with colors indexed to unique values.
    """
    mask = cv2.imread(path, cv2.IMREAD_COLOR)

    if mask is not None and plot_mask:
        plt.imshow(mask)
        plt.show()

    return mask


def load_hypercubes(n_max_cubes=None, plot_hc=False, plot_mask=False, folder='', baseline_class_idx = 0,
                    color_dict = None):
    """
    Loads all hypercubes from the given folder.
    """
    cube_paths = glob.glob(folder + 'raw*rf' + paths.hc_extension)
    cubes = []
    max_class_idx = 0
    my_color_dict = color_dict

    for idx, path in enumerate(cube_paths):
        file_name = path[0: len(path) - len(paths.hc_extension)]
        if len(file_name) == 0:
            continue

        print('Reading ' + path + ' ...')
        hc_numpy, hc_bands = __load_hc(path, plot_hc=plot_hc)
        class_mask = __load_mask(file_name + paths.class_mask_extension, plot_mask=plot_mask)

        if hc_numpy is not None and class_mask is not None:
            hc = Hypercube(hc_numpy, class_mask, hc_bands, path, baseline_class_idx=baseline_class_idx,
                           color_dict=my_color_dict)
            hc.filter_wl(hc_bands[25], hc_bands[-25])
            max_class_idx = int(max(max_class_idx, np.max(hc.get_labels())))
            my_color_dict = hc.get_color_dict().copy()

            cubes.append(hc)

        if idx >= (n_max_cubes - 1):
            break

    return cubes, max_class_idx, my_color_dict


def __load_umat(hc_numpy, class_mask, path, plot_hc=False, plot_mask=False):
    # Get class with the most pixels
    values, counts = np.unique(class_mask, return_counts=True)
    max_class = values[np.argmax(counts)]

    # Print
    print('Max class: ' + str(max_class))
    for (label, count) in zip(values, counts):
        # Count pixels of each class
        print('Class ' + str(label) + ': ' + str(count))

    hc = Hypercube(hc_numpy, class_mask, None, path, null_class_idx=max_class)

    if plot_hc:
        plt.imshow(hc_numpy[:, :, hc_numpy.shape[2] // 2])
        plt.show()

    if plot_mask:
        plt.imshow(class_mask)
        plt.show()

    return hc, max_class


def load_pavia_umat(plot_hc=False, plot_mask=False):
    """
    Loads the Pavia Umat dataset.
    """
    # Load pavia umat as numpy array
    pavia_umat = sio.loadmat(paths.pavia_umat_path)['paviaU']
    hc_numpy = np.array(pavia_umat, dtype=np.float32)
    pavia_umat_gt = sio.loadmat(paths.pavia_umat_mask_path)
    class_mask = np.array(pavia_umat_gt['paviaU_gt']) - 1

    return __load_umat(hc_numpy, class_mask, paths.pavia_umat_path, plot_hc=plot_hc, plot_mask=plot_mask)


def load_pavia_centre_umat(plot_hc=False, plot_mask=False):
    """
    Loads the Pavia Umat dataset.
    """
    # Load pavia umat as numpy array
    pavia_umat = sio.loadmat(paths.pavia_centre_umat_path)['pavia']
    hc_numpy = np.array(pavia_umat, dtype=np.float32)
    pavia_umat_gt = sio.loadmat(paths.pavia_centre_umat_mask_path)
    class_mask = np.array(pavia_umat_gt['pavia_gt']) - 1

    return __load_umat(hc_numpy, class_mask, paths.pavia_centre_umat_path, plot_hc=plot_hc, plot_mask=plot_mask)


def load_indian_pines_umat(plot_hc=False, plot_mask=False):
    """
    Loads the Indian Pines Umat dataset.
    """
    # Load pavia umat as numpy array
    # print(sio.loadmat(paths.indian_pines_umat_path))
    indian_pines_umat = sio.loadmat(paths.indian_pines_umat_path)['indian_pines_corrected']
    hc_numpy = np.array(indian_pines_umat, dtype=np.float32)
    indian_pines_umat_gt = sio.loadmat(paths.indian_pines_umat_mask_path)
    class_mask = np.array(indian_pines_umat_gt['indian_pines_gt']) - 1

    return __load_umat(hc_numpy, class_mask, paths.indian_pines_umat_path, plot_hc=plot_hc, plot_mask=plot_mask)


def load_salinas_umat(plot_hc=False, plot_mask=False):
    """
    Loads the Salinas Umat dataset.
    """
    # Load pavia umat as numpy array
    salinas_umat = sio.loadmat(paths.salinas_umat_path)['salinas_corrected']
    hc_numpy = np.array(salinas_umat, dtype=np.float32)
    salinas_umat_gt = sio.loadmat(paths.salinas_umat_mask_path)
    class_mask = np.array(salinas_umat_gt['salinas_gt']) - 1

    return __load_umat(hc_numpy, class_mask, paths.salinas_umat_path, plot_hc=plot_hc, plot_mask=plot_mask)


def load_salinas_a_umat(plot_hc=False, plot_mask=False):
    """
    Loads the Salinas Umat dataset.
    """
    # Load pavia umat as numpy array
    salinas_umat = sio.loadmat(paths.salinas_a_umat_path)['salinasA_corrected']
    hc_numpy = np.array(salinas_umat, dtype=np.float32)
    salinas_umat_gt = sio.loadmat(paths.salinas_a_umat_mask_path)
    class_mask = np.array(salinas_umat_gt['salinasA_gt']) - 1

    return __load_umat(hc_numpy, class_mask, paths.salinas_a_umat_path, plot_hc=plot_hc, plot_mask=plot_mask)

def load_umat(folder, plot_hc=False, plot_mask=False):
    # Search files with extension .gt
    files = [f for f in os.listdir(folder) if f.endswith('.mat')]

    for file in files:
        if file.endswith('_gt.mat'):
            continue
        else:
            # Remove extension from file name
            file_name = file.split('.')[0]
            class_mask_file = file_name + '_gt.mat'

            if class_mask_file in files:
                hc_numpy = sio.loadmat(os.path.join(folder, file))[file_name]
                hc_numpy = np.array(hc_numpy, dtype=np.float32)

                class_mask = sio.loadmat(os.path.join(folder, class_mask_file))[file_name + '_gt']
                class_mask = np.array(class_mask, dtype=np.int32)

                return __load_umat(hc_numpy, class_mask, file, plot_hc=plot_hc, plot_mask=plot_mask)

    return None