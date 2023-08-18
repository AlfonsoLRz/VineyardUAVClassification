import numpy as np
from skimage.metrics import structural_similarity as compare_ssim


def computeGFC(groundTruth, recovered):
    """
    Compute Goodness-of-Fit Coefficient (GFC) between the recovered and the
    corresponding ground-truth image
    """
    GFCn = np.sum(np.multiply(groundTruth, recovered))
    GFCd = np.multiply(np.sqrt(np.sum(np.power(groundTruth, 2))),
                       np.sqrt(np.sum(np.power(recovered, 2))))
    GFC = np.divide(GFCn, GFCd)
    return np.mean(GFC)


def computeMSSIM(groundTruth, recovered):
    """
    Compute Mean Structural SImilarity Measure (MSSIM) between
    the recovered and the corresponding ground-truth image
    """
    # to get SSIM put full = True to get values instead of mean
    return compare_ssim(groundTruth, recovered, multichannel=True)
