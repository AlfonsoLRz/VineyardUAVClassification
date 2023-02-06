from alive_progress import alive_bar
from enum import Enum
import hypercube as hc
import randomness
from randomness import *
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
    Sousao = 8
    TourigaNacional = 2
    TourigaFrancesa = 7
    Alicante = 3
    TourigaFemea = 4
    Barroca = 6
    TintaRoriz = 5


class WhiteVineyardLabel(Enum):
    """
    Enum for the different white wine labels.
    """
    Unknown1 = 1
    Boal = 2
    Unknown2 = 3
    Unknown3 = 4
    CodegaDoLadinho = 5
    Unknown4 = 6
    MoscatelGalego = 7
    NascatelGalegoRoixo = 8
    AritoDoDouro = 9
    Cercial = 10
    MalvasiaFina = 11
    Unknown5 = 12
    Unknown6 = 13


class HypercubeSet:
    def __init__(self, hc_array):
        """
        Initializes a hypercube set.
        :param hc_array: Array of loaded hypercubes.
        """
        self._hypercube = None
        self._mask = None
        self._hypercube_shapes = []
        self._bands = None
        self._train_indices = None
        self._test_indices = None
        self._remove_ground_indices = None

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

        del hc_array

    def compose_swath_evaluation(self, y_label, prediction, patch_size):
        if patch_size % 2 == 0:
            patch_size += 1
        swath_shape = self._hypercube_shapes[0]
        swath_shape = (swath_shape[0] - patch_size, swath_shape[1] - patch_size)

        shaped_label = np.reshape(y_label, swath_shape)
        shaped_prediction = np.reshape(prediction, swath_shape)
        diff = np.abs(shaped_label - shaped_prediction)
        diff[diff > 0] = 1

        return diff

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

    def obtain_train_indices(self, test_percentage, patch_size, patch_overlapping):
        """
        Obtains the train indices.
        :param test_percentage: Percentage of the data to be used as test data.
        """
        num_pixels = self._mask.shape[0] * self._mask.shape[1]
        # Remove ground indices from the overall set
        available_indices = np.setdiff1d(np.arange(0, num_pixels, step=patch_size - patch_overlapping),
                                         self._remove_ground_indices)
        num_pixels = len(available_indices)

        self._train_indices = np.random.choice(available_indices, int(num_pixels * (1.0 - test_percentage)),
                                               replace=False)
        self._test_indices = np.setdiff1d(available_indices, self._train_indices)

    def obtain_ground_labels(self):
        """
        Removes the ground labels from the hypercube set.
        """
        # Maximum number of vegetation samples
        max_vegetation_samples = 0
        for label in range(1, self.get_num_classes()):
            max_vegetation_samples = max(max_vegetation_samples, np.sum(self._mask == label))

        # Flatten mask
        mask = np.reshape(self._mask, self._mask.shape[0] * self._mask.shape[1])
        # Ground indices
        ground_indices = np.where(mask == 0)[0]
        del mask
        num_removable_indices = len(ground_indices) - max_vegetation_samples
        np.random.seed(randomness.random_seed)
        self._remove_ground_indices = np.random.choice(ground_indices, num_removable_indices, replace=False)

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

    @staticmethod
    def reduce_layers(hypercube, mask, n_layers=30, selection_method=LayerSelectionMethod.PCA):
        """
        Reduces the number of layers in the hypercube.
        """
        reduction = None

        if selection_method == LayerSelectionMethod.PCA:
            reduction = PCA(n_components=n_layers, random_state=random_seed)
        elif selection_method == LayerSelectionMethod.FACTOR_ANALYSIS:
            reduction = FactorAnalysis(n_components=n_layers, random_state=random_seed)
        elif selection_method == LayerSelectionMethod.SVD:
            reduction = TruncatedSVD(n_components=n_layers, random_state=random_seed)
        elif selection_method == LayerSelectionMethod.NMF:
            reduction = NMF(n_components=n_layers, random_state=random_seed)
        elif selection_method == LayerSelectionMethod.LDA:
            reduction = LDA(n_components=n_layers)

        if selection_method == LayerSelectionMethod.LDA:
            reduction.fit(hypercube, y=mask)
        else:
            reduction.fit(hypercube)

        return reduction

    def render_class_profile(self):
        """
        Renders the distribution of vineyard labels.
        """
        threed_shape = self._hypercube.shape
        self._hypercube, self._mask = self.flatten()
        render_hc_spectrum_label(self._hypercube, self._mask)
        self._to_3d(threed_shape)

    def split_train(self, patch_size, max_train_samples=None):
        """
        Splits the hypercube set into train and test sets.
        """
        if max_train_samples is None:
            max_train_samples = self._train_indices.shape[0]
        else:
            max_train_samples = min(max_train_samples, self._train_indices.shape[0])

        train_indices = self._train_indices[:max_train_samples]
        self._train_indices = self._train_indices[max_train_samples:]

        return self.__split_indices(train_indices, patch_size, self._hypercube.shape)

    def split_swath(self, patch_size, patch_id=0, limit=None, offset=0):
        swath_shape = self._hypercube_shapes[patch_id]
        num_pixels = swath_shape[0] * swath_shape[1]

        if patch_id < len(self._hypercube_shapes) and offset < num_pixels:
            if limit is not None:
                num_pixels = min(min(num_pixels, limit), num_pixels - offset)
            available_indices = np.arange(num_pixels * patch_id + offset, num_pixels * patch_id + offset + num_pixels,
                                          step=1)

            return self.__split_indices(available_indices, patch_size, swath_shape)

        return None, None

    def split_test(self, patch_size):
        """
        Splits the hypercube into test patches.
        """
        return self.__split_indices(self._test_indices, patch_size, self._hypercube.shape)

    def __split_indices(self, indices, patch_size, big_hypercube_shape):
        patch = []
        label = []
        hypercube_shape = self._hypercube_shapes[0]
        num_train_samples = indices.shape[0]
        half_patch_size = patch_size // 2

        with alive_bar(num_train_samples, force_tty=True) as bar:
            for index in indices:
                y, x = index // big_hypercube_shape[1], index % big_hypercube_shape[1]
                hypercube_index = int(x // hypercube_shape[1])
                base_hypercube_x = hypercube_index * hypercube_shape[1]
                x -= base_hypercube_x

                if (x - half_patch_size) >= 0 and (x + half_patch_size + 1) < hypercube_shape[1] and \
                        (y - half_patch_size) >= 0 and (y + half_patch_size + 1) < hypercube_shape[0]:
                    x += base_hypercube_x
                    patch.append(self._hypercube[y - half_patch_size:y + half_patch_size + 1,
                                 x - half_patch_size:x + half_patch_size + 1, :])
                    label.append(self._mask[y - half_patch_size:y + half_patch_size + 1,
                                 x - half_patch_size:x + half_patch_size + 1])

                bar()

        return np.asarray(patch), np.asarray(label)

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

        with alive_bar(2, force_tty=True) as bar:
            reduction = self.reduce_layers(n_layers=num_features, selection_method=selection_method,
                                           hypercube=self._hypercube[self._train_indices],
                                           mask=self._mask[self._train_indices])
            self._hypercube = reduction.transform(self._hypercube)

            bar()

            if standardize:
                standard_scaler = StandardScaler()
                standard_scaler.fit(self._hypercube[self._train_indices])
                self._hypercube = standard_scaler.transform(self._hypercube)

            bar()

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
