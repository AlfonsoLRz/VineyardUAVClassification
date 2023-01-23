from enum import Enum

import hypercube as hc
from rendering import *
from sklearn.decomposition import FactorAnalysis, NMF, PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


class LayerSelectionMethod(Enum):
    """
    Enum for the different layer selection methods.
    """
    PCA = 0
    FACTOR_ANALYSIS = 1
    SVD = 2
    NMF = 3
    LDA = 4


class RedVineyardLabel(Enum):
    """
    Enum for the different red wine labels.
    """
    Alvarelhao = 1
    Sousao = 2
    TourigaNacional = 3
    TourigaFrancesa = 4
    Alicante = 5
    TourigaFemea = 6
    Barroca = 7
    TintaRoriz = 8


class WhiteVineyardLabel(Enum):
    """
    Enum for the different white wine labels.
    """
    UNKNOWN_1 = 1
    BOAL = 2
    UNKNOWN_2 = 3
    UNKNOWN_3 = 4
    CODEGA_DO_LADINHO = 5
    UNKNOWN_4 = 6
    MOSCATEL_GALEGO = 7
    NASCATEL_GALECO_ROXO = 8
    ARITO_DO_DOURO = 9
    CERCIAL = 10
    MALVASIA_FINA = 11
    UNKNOWN_5 = 12
    UNKNOWN_6 = 13


class HypercubeSet:
    def __init__(self, hc_array):
        """
        Initializes a hypercube set.
        :param hc_array: Array of loaded hypercubes.
        """
        self._hypercube = None
        self._mask = None
        self._hypercube_instances = hc_array
        self._hypercube_shapes = []
        self._bands = None

        if hc_array is not None and len(hc_array) > 0:
            self._bands = hc_array[0].get_bands()

            # Determine min. height
            min_height = hc_array[0].get_hypercube().shape[0]
            for hypercube_idx in range(1, len(hc_array)):
                if hc_array[hypercube_idx].get_hypercube().shape[0] < min_height:
                    min_height = hc_array[hypercube_idx].get_hypercube().shape[0]

            for hypercube in hc_array:
                if self._hypercube is not None:
                    self._hypercube = np.concatenate((self._hypercube, hypercube.get_hypercube()[:min_height, :, :]),
                                                     axis=1)
                    self._mask = np.concatenate((self._mask, hypercube.get_class_mask()[:min_height, :]), axis=1)
                else:
                    self._hypercube = hypercube.get_hypercube()[:min_height, :, :]
                    self._mask = hypercube.get_class_mask()[:min_height, :]

                self._hypercube_shapes.append((min_height, hypercube.get_shape()[1], hypercube.get_shape()[2]))

            self._mask = HypercubeSet._to_id_image(self._mask)

    def flatten(self):
        """
        Turns a 3D hypercube into a 2D vector for preprocessing such as matrix factorization.
        :return: Flattened hypercube and mask.
        """
        return np.reshape(self._hypercube,
                          (self._hypercube.shape[0] * self._hypercube.shape[1], self._hypercube.shape[2])), \
               np.reshape(self._mask, self._mask.shape[0] * self._mask.shape[1])

    def get_num_classes(self):
        """
        :return: Number of classes in the hypercube set.
        """
        return int(np.max(self._mask)) + 1

    def get_num_hypercubes(self):
        """
        :return: Number of hypercubes in the set.
        """
        return len(self._hypercube_shapes)

    def get_rgb_indices(self):
        """
        :return: Indices of the RGB bands.
        """
        if self._hypercube is not None:
            return hc.Hypercube.get_rgb_indices(self._hypercube_instances[0])

        return None

    def get_shape(self):
        """
        :return: Shape of the hypercube set.
        """
        return self._hypercube.shape

    def get_vegetation_variance(self):
        """
        Gets the vegetation variance from the hypercube.
        :return: Vegetation variance.
        """
        variance = 0
        num_classes = self.get_num_classes()

        for class_idx in range(1, num_classes):
            variance += np.var(self._hypercube[self._mask == class_idx, :])

        return variance / (num_classes - 1)

    def plot(self, plot_hc=True, plot_mask=True):
        """
        Plots the hypercube set.
        """
        if plot_hc:
            plt.imshow(self._hypercube[:, :, 0])
            plt.show()

        if plot_mask:
            plt.imshow(self._mask)
            plt.show()

    def print_metadata(self, title=''):
        """
        Prints the metadata of the hypercube set.
        """
        print(title + 'Min: {}, Max: {}, Size: {}'.format(np.min(self._hypercube), np.max(self._hypercube),
                                                          self._hypercube.shape))

    def print_num_samples_per_label(self, red=True):
        """
        Prints the number of samples per label.
        """
        for label in range(1, self.get_num_classes()):
            if red:
                print('Label {}: {}'.format(RedVineyardLabel(label).name, np.sum(self._mask == label)))
            else:
                print('Label {}: {}'.format(WhiteVineyardLabel(label).name, np.sum(self._mask == label)))

    def reduce_layers(self, n_layers=30, selection_method=LayerSelectionMethod.PCA):
        """
        Reduces the number of layers in the hypercube.
        """
        reduction = None

        if selection_method == LayerSelectionMethod.PCA:
            reduction = PCA(n_components=n_layers, random_state=random_state)
        elif selection_method == LayerSelectionMethod.FACTOR_ANALYSIS:
            reduction = FactorAnalysis(n_components=n_layers, random_state=random_state)
        elif selection_method == LayerSelectionMethod.SVD:
            reduction = TruncatedSVD(n_components=n_layers, random_state=random_state)
        elif selection_method == LayerSelectionMethod.NMF:
            reduction = NMF(n_components=n_layers, random_state=random_state)
        elif selection_method == LayerSelectionMethod.LDA:
            reduction = LDA(n_components=n_layers)

        if selection_method == LayerSelectionMethod.LDA:
            self._hypercube = reduction.fit_transform(hc, y=self._mask)
        else:
            self._hypercube = reduction.fit_transform(hc)

        return reduction

    def render_class_profile(self):
        """
        Renders the distribution of vineyard labels.
        """
        threed_shape = self._hypercube.shape
        self._hypercube, self._mask = self.flatten()
        render_hc_spectrum_class(self._hypercube, self._mask)
        self._to3D(threed_shape)

    def split_all(self, chunk_size, overlapping):
        """
        Splits the whole hypercube set into chunks.
        """
        hc_indices = [i for i in range(self.get_num_hypercubes())]
        return self.split(hc_indices, chunk_size, overlapping)

    def split(self, hc_ids, chunk_size, overlapping):
        """
        Splits the hypercubes selected by hc_ids into chunks.
        """
        jump = chunk_size - overlapping
        chunk = []
        chunk_label = []

        for hc_id in hc_ids:
            hc_shape = self._hypercube_shapes[hc_id]
            x = hc_shape[1] * hc_id
            boundary_x = x + hc_shape[1]

            while x + chunk_size < boundary_x:
                y = 0

                while y + chunk_size < hc_shape[0]:
                    chunk.append(self._hypercube[int(y):int(y + chunk_size), int(x):int(x + chunk_size), :])
                    chunk_label.append(self._mask[int(y):int(y + chunk_size), int(x):int(x + chunk_size)])

                    y += jump

                x += jump

        return np.array(chunk), np.array(chunk_label)

    def standardize(self, num_features=30, standardize=True, selection_method=LayerSelectionMethod.FACTOR_ANALYSIS):
        """
        Preprocessing of UAV data into relevant data for DL.
        """
        threed_shape = self._hypercube.shape
        self._hypercube, self._mask = self.flatten()

        self.reduce_layers(n_layers=num_features, selection_method=selection_method)
        if standardize:
            self._hypercube = StandardScaler().fit_transform(self._hypercube)

        threed_shape = (threed_shape[0], threed_shape[1], self._hypercube.shape[1])
        self._to_3d(threed_shape)

    def _to_3d(self, shape_3d):
        """
        Turns a 2D vector into a 3D hypercube.
        """
        self._hypercube = np.reshape(self._hypercube, (shape_3d[0], shape_3d[1], shape_3d[2]))
        self._mask = np.reshape(self._mask, (shape_3d[0], shape_3d[1]))

    @staticmethod
    def _to_id_image(img):
        h = img.shape[0]
        w = img.shape[1]
        id_image = np.zeros(shape=(h, w))
        color_dict = {(0, 0, 0): 0}

        for y in range(0, h):
            for x in range(0, w):
                color = (int(img[y, x, 0]), int(img[y, x, 1]), int(img[y, x, 2]))
                if color not in color_dict:
                    color_dict[color] = len(color_dict)

                id_image[y, x] = color_dict[color]

        return id_image
