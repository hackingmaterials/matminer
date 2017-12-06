import numpy as np

@staticmethod
def laplacian_kernel(arr0, arr1, SIGMA):
    """
    Returns a Laplacian kernel of the two arrays
    for use in KRR or other regressions using the
    kernel trick.
    """
    diff = arr0 - arr1
    return np.exp(-np.linalg.norm(diff.A1, ord=1) / SIGMA)


@staticmethod
def gaussian_kernel(arr0, arr1, SIGMA):
    """
    Returns a Gaussian kernel of the two arrays
    for use in KRR or other regressions using the
    kernel trick.
    """
    diff = arr0 - arr1
    return np.exp(-np.linalg.norm(diff.A1, ord=2) ** 2 / 2 / SIGMA ** 2)