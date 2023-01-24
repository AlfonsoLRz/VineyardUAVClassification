import cv2
import glob
from hypercube import Hypercube
import matplotlib.pyplot as plt
import paths
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


def load_hypercubes(n_max_cubes=None, plot_hc=False, plot_mask=False, additional_root='', selected_cubes=[]):
    """
    Loads all hypercubes from the given folder.
    """
    cube_paths = glob.glob(additional_root + paths.folder_path + 'raw*rf' + paths.hc_extension)
    cubes = []

    for idx, path in enumerate(cube_paths):
        if len(selected_cubes) > 0 and idx not in selected_cubes:
            continue

        file_name = path[0: len(path) - len(paths.hc_extension)]
        if len(file_name) == 0:
            continue

        print('Reading ' + path + ' ...')
        hc_numpy, hc_bands = __load_hc(path, plot_hc=plot_hc)
        class_mask = __load_mask(file_name + paths.class_mask_extension, plot_mask=plot_mask)

        if hc_numpy is not None and class_mask is not None:
            hc = Hypercube(hc_numpy, class_mask, hc_bands, path)
            hc.filter_wl(450, 950)
            cubes.append(hc)

        if idx >= n_max_cubes - 1:
            break

    return cubes