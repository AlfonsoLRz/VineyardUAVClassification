import numpy as np


def computeMSE(groundTruth, recovered):
    """
    Compute Mean Square Error (MSE) between the recovered and the
    corresponding ground-truth image
    """
    square_diff = np.power(groundTruth - recovered, 2)
    mse = np.mean(square_diff)

    return mse


def computeRMSE(groundTruth, recovered):
    """
    Compute Root Mean Square Error (RMSE) between the recovered and the
    corresponding ground-truth image
    """
    square_diff = np.power(groundTruth - recovered, 2)
    rmse = np.sqrt(np.mean(square_diff))

    return rmse


def computeMRAE(groundTruth, recovered):
    """
    Compute Mean Relative Absolute Error (MRAE) between the recovered and the
    corresponding ground-truth image
    """
    difference = np.abs(groundTruth - recovered) / (groundTruth + 1e-15)
    mrae = np.mean(difference)

    return mrae


def computePSNR(groundTruth, recovered):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between the recovered and the
    corresponding ground-truth image
    """
    psnr = 20 * np.log10(1 / np.sqrt(computeMSE(groundTruth, recovered) + 1e-15))

    return psnr
