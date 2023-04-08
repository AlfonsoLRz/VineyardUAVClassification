import numpy as np


def computeAPPSA(groundTruth, recovered):
    """
    Compute Average Per-Pixel Spectral Angle (APPA) between the recovered and
    the corresponding ground-truth image
    """
    nom = np.sum(groundTruth * recovered)
    denom = np.linalg.norm(groundTruth) * np.linalg.norm(recovered)
    cos = np.where((nom / (denom + 1e-15)) > 1, 1, (nom / (denom + 1e-15)))
    appsa = np.arccos(cos)

    return np.sum(appsa)


def computeED(groundTruth, recovered):
    """
    Compute Euclidean Distance (ED) between the recovered and
    the corresponding ground-truth image
    """
    ed = np.sqrt(np.sum(np.power(groundTruth - recovered, 2)))
    return ed


def computeMAngE(groundTruth, recovered):
    """
    Compute Mean Angular Error (MangE) between the recovered and
    the corresponding ground-truth image
    """
    inner_product = np.sum(groundTruth * recovered)   # DIM_DATA,
    normalized_inner_product = inner_product / (np.linalg.norm(groundTruth) + 1e-15) / (np.linalg.norm(recovered) + 1e-15)

    mangE = np.arccos(normalized_inner_product) * 180 / np.pi
    mangE *= np.sqrt(np.mean(groundTruth))

    return mangE


def computeSAM(groundTruth, recovered):
    """
    Compute Spectral Angle Mapper (SAM) between the recovered and
    the corresponding ground-truth image
    """
    nom = np.sum(np.multiply(groundTruth, recovered))
    denom1 = np.sqrt(np.sum(np.power(groundTruth, 2)))
    denom2 = np.sqrt(np.sum(np.power(recovered, 2)))
    sam = np.arccos(np.divide(nom, np.multiply(denom1, denom2)))
    return sam
