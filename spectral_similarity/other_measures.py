import numpy as np


def computeMSID(groundTruth, recovered):
    """
    Compute Mean Spectral Information Divergence (MSID) between
    the recovered and the corresponding ground-truth image
    """
    sumRC = np.sum(recovered)
    sumGT = np.sum(groundTruth)

    logRC = np.log((recovered / sumRC + 1e-15) / (groundTruth / sumGT + 1e-15))
    logGT = np.log((groundTruth / sumGT + 1e-15) / (recovered / sumRC + 1e-15))

    err = abs(np.sum(recovered * logRC) + np.sum(groundTruth * logGT))

    return err
