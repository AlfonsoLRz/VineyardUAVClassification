import cv2
import glob

import rendering
from hypercube import Hypercube
import matplotlib.pyplot as plt
import numpy as np
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


def load_hypercubes(n_max_cubes=None, plot_hc=False, plot_mask=False, folder='', baseline_class_idx = 0):
    """
    Loads all hypercubes from the given folder.
    """
    cube_paths = glob.glob(folder + 'raw*rf' + paths.hc_extension)
    cubes = []
    max_class_idx = 0

    for idx, path in enumerate(cube_paths):
        file_name = path[0: len(path) - len(paths.hc_extension)]
        if len(file_name) == 0:
            continue

        print('Reading ' + path + ' ...')
        hc_numpy, hc_bands = __load_hc(path, plot_hc=plot_hc)
        class_mask = __load_mask(file_name + paths.class_mask_extension, plot_mask=plot_mask)

        if hc_numpy is not None and class_mask is not None:
            hc = Hypercube(hc_numpy, class_mask, hc_bands, path, baseline_class_idx=baseline_class_idx)
            hc.filter_wl(hc_bands[25], hc_bands[-25])
            max_class_idx = int(max(max_class_idx, np.max(hc.get_labels())))

            cubes.append(hc)

            print(hc.get_labels())

        if idx >= (n_max_cubes - 1):
            break

    return cubes, max_class_idx


def load_umat(hc_numpy, class_mask, path, plot_hc=False, plot_mask=False):
    hc = Hypercube(hc_numpy, class_mask, None, path)

    if plot_hc:
        plt.imshow(hc_numpy[:, :, hc_numpy.shape[2] // 2])
        plt.show()

    if plot_mask:
        plt.imshow(class_mask)
        plt.show()

    # Change value 255 to 0
    values = np.unique(class_mask)
    class_mask[class_mask == 255] = len(values) - 1
    values = np.unique(class_mask)

    for v in values:
        # Count pixels of each class
        print('Class ' + str(v) + ': ' + str(np.count_nonzero(class_mask == v)))

    return hc


def load_pavia_umat(plot_hc=False, plot_mask=False):
    """
    Loads the Pavia Umat dataset.
    """
    # Load pavia umat as numpy array
    pavia_umat = sio.loadmat(paths.pavia_umat_path)['paviaU']
    hc_numpy = np.array(pavia_umat, dtype=np.float32)
    pavia_umat_gt = sio.loadmat(paths.pavia_umat_mask_path)
    class_mask = np.array(pavia_umat_gt['paviaU_gt']) - 1

    return load_umat(hc_numpy, class_mask, paths.pavia_umat_path, plot_hc=plot_hc, plot_mask=plot_mask)


def load_pavia_centre_umat(plot_hc=False, plot_mask=False):
    """
    Loads the Pavia Umat dataset.
    """
    # Load pavia umat as numpy array
    pavia_umat = sio.loadmat(paths.pavia_centre_umat_path)['pavia']
    hc_numpy = np.array(pavia_umat, dtype=np.float32)
    pavia_umat_gt = sio.loadmat(paths.pavia_centre_umat_mask_path)
    class_mask = np.array(pavia_umat_gt['pavia_gt']) - 1

    return load_umat(hc_numpy, class_mask, paths.pavia_centre_umat_path, plot_hc=plot_hc, plot_mask=plot_mask)


def load_indian_pines_umat(plot_hc=False, plot_mask=False):
    """
    Loads the Indian Pines Umat dataset.
    """
    # Load pavia umat as numpy array
    print(sio.loadmat(paths.indian_pines_umat_path))
    indian_pines_umat = sio.loadmat(paths.indian_pines_umat_path)['indian_pines_corrected']
    hc_numpy = np.array(indian_pines_umat, dtype=np.float32)
    indian_pines_umat_gt = sio.loadmat(paths.indian_pines_umat_mask_path)
    class_mask = np.array(indian_pines_umat_gt['indian_pines_gt']) - 1

    return load_umat(hc_numpy, class_mask, paths.indian_pines_umat_path, plot_hc=plot_hc, plot_mask=plot_mask)


def load_salinas_umat(plot_hc=False, plot_mask=False):
    """
    Loads the Salinas Umat dataset.
    """
    # Load pavia umat as numpy array
    salinas_umat = sio.loadmat(paths.salinas_umat_path)['salinas_corrected']
    hc_numpy = np.array(salinas_umat, dtype=np.float32)
    salinas_umat_gt = sio.loadmat(paths.salinas_umat_mask_path)
    class_mask = np.array(salinas_umat_gt['salinas_gt']) - 1

    return load_umat(hc_numpy, class_mask, paths.salinas_umat_path, plot_hc=plot_hc, plot_mask=plot_mask)


def load_salinas_a_umat(plot_hc=False, plot_mask=False):
    """
    Loads the Salinas Umat dataset.
    """
    # Load pavia umat as numpy array
    salinas_umat = sio.loadmat(paths.salinas_a_umat_path)['salinasA_corrected']
    hc_numpy = np.array(salinas_umat, dtype=np.float32)
    salinas_umat_gt = sio.loadmat(paths.salinas_a_umat_mask_path)
    class_mask = np.array(salinas_umat_gt['salinasA_gt']) - 1

    return load_umat(hc_numpy, class_mask, paths.salinas_a_umat_path, plot_hc=plot_hc, plot_mask=plot_mask)
