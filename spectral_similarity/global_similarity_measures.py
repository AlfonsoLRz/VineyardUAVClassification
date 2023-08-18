import enum
from joblib import Parallel, delayed
from spectral_similarity.other_measures import *
import spectral_similarity
import spectral_similarity.angular_measures
import spectral_similarity.colorimetric_measures
import spectral_similarity.error_measures
import spectral_similarity.other_measures
import spectral_similarity.similarity_measures
import spectral_similarity.utils


class SimilarityMethod(enum.IntEnum):
    """
    Enum for the similarity methods.
    """
    SAM = 1
    MangE = 2
    APPSA = 3
    GFC = 4
    MSSIM = 5
    MSE = 6
    RMSE = 7
    MRAE = 8
    PSNR = 9
    MSID = 10
    ED = 11


def compute_similarity(X, y, method, rendering=True, multicore=False, aggregate=True):
    similarity_func = None
    if method == SimilarityMethod.SAM:
        similarity_func = 'spectral_similarity.angular_measures.computeSAM'
    elif method == SimilarityMethod.MangE:
        similarity_func = 'spectral_similarity.angular_measures.computeMAngE'
    elif method == SimilarityMethod.APPSA:
        similarity_func = 'spectral_similarity.angular_measures.computeAPPSA'
    elif method == SimilarityMethod.GFC:
        similarity_func = 'spectral_similarity.similarity_measures.computeGFC'
    elif method == SimilarityMethod.MSSIM:
        similarity_func = 'spectral_similarity.similarity_measures.computeMSSIM'
    elif method == SimilarityMethod.MSE:
        similarity_func = 'spectral_similarity.error_measures.computeMSE'
    elif method == SimilarityMethod.RMSE:
        similarity_func = 'spectral_similarity.error_measures.computeRMSE'
    elif method == SimilarityMethod.MRAE:
        similarity_func = 'spectral_similarity.error_measures.computeMRAE'
    elif method == SimilarityMethod.PSNR:
        similarity_func = 'spectral_similarity.error_measures.computePSNR'
    elif method == SimilarityMethod.MSID:
        similarity_func = 'spectral_similarity.other_measures.computeMSID'
    elif method == SimilarityMethod.ED:
        similarity_func = 'spectral_similarity.angular_measures.computeED'


    print(similarity_func)

    labels = np.unique(y)
    mean_signatures = []
    for label in labels:
        mean_signatures.append(np.mean(X[y == label], axis=0))

    if multicore:
        similar_index = Parallel(n_jobs=30, backend='multiprocessing')\
            (delayed(__evaluate_similarity)(similarity_func, X[sample_idx], mean_signatures[int(y[sample_idx])])
             for sample_idx in range(X.shape[0]))
    else:
        similar_index = np.zeros(X.shape[0])
        for sample_idx in range(X.shape[0]):
            try:
                similar_index[sample_idx] = __evaluate_similarity(similarity_func, X[sample_idx],
                                                                  mean_signatures[int(y[sample_idx])])
            except ZeroDivisionError:
                similar_index[sample_idx] = 0

    return similar_index
    # if aggregate:
    #     labels = np.unique(hc_labels[non_soil_indices])
    #     for label in labels:
    #         label_indices = np.where(hc_labels == label)[0]
    #         assigned_indices, counts = np.unique(similar_index_img[label_indices], return_counts=True)
    #         label = assigned_indices[np.argmax(counts)]
    #         similar_index_img[label_indices] = label

    # hc_shape = hc_set.get_shape()
    # similar_index_img_reshaped = np.reshape(similar_index_img, (hc_shape[0], hc_shape[1]))
    #
    # if rendering:
    #     plt.imshow(similar_index_img_reshaped)
    #     plt.axis('off')
    #     plt.show()
    #
    #     dataset_splitter.render_mask_histogram(similar_index)
    #     if aggregate:
    #         dataset_splitter.render_mask_histogram(similar_index_img[non_soil_indices])


def __evaluate_similarity(similarity_func, sample, example):
    similarity_stack = np.zeros(sample.shape[0])
    for i in range(sample.shape[0]):
        similarity_stack[i] = eval(similarity_func)(sample, example)

    return np.argmin(similarity_stack) + 1
