from __future__ import division

import scipy

"""
General methods for computing property statistics from a list of values
"""

import numpy as np
from scipy import stats

from six import string_types


class PropertyStats(object):
    """This class contains statistical operations that are commonly employed
    when computing features.

    The primary way for interacting with this class is to call the
    ``calc_stat`` function, which takes the name of the statistic you would
    like to compute and the weights/values of data to be assessed. For example,
    computing the mean of a list looks like::

        x = [1, 2, 3]
        PropertyStats.calc_stat(x, 'mean') # Result is 2
        PropertyStats.calc_stat(x, 'mean', weights=[0, 0, 1]) # Result is 3

    Some of the statistics functions take options (e.g., Holder means). You can
    pass them to the the statistics functions by adding them after the name and
    two colons. For example, the 0th Holder mean would be::

        PropertyStats.calc_stat(x, 'holder_mean::0')

    You can, of course, call the statistical functions directly. All take at
    least two arguments.  The first is the data being assessed and the second,
    optional, argument is the weights.
    """

    @staticmethod
    def calc_stat(data_lst, stat, weights=None):
        """
        Compute a property statistic

        Args:
            data_lst (list of floats): list of values
            stat (str) - Name of property to be compute. If there are arguments to the statistics function, these
             should be added after the name and separated by two colons. For example, the 2nd Holder mean would
             be "holder_mean::2"
            weights (list of floats): (Optional) weights for each element in data_lst
        Returns:
            float - Desired statistic
        """
        statistics = stat.split("::")
        return getattr(PropertyStats, statistics[0])(data_lst, weights, *statistics[1:])

    @staticmethod
    def minimum(data_lst, weights=None):
        """Minimum value in a list

        Args:
            data_lst (list of floats): List of values to be assessed
            weights: (ignored)
        Returns:
            minimum value
        """
        return min(data_lst) if not np.any(np.isnan(data_lst)) else float("nan")

    @staticmethod
    def maximum(data_lst, weights=None):
        """Maximum value in a list

        Args:
            data_lst (list of floats): List of values to be assessed
            weights: (ignored)
        Returns:
            maximum value
        """
        return max(data_lst) if not np.any(np.isnan(data_lst)) else float("nan")

    @staticmethod
    def range(data_lst, weights=None):
        """Range of a list

        Args:
            data_lst (list of floats): List of values to be assessed
            weights: (ignored)
        Returns:
            range
        """
        return (max(data_lst) - min(data_lst)) if not np.any(np.isnan(data_lst)) \
            else float("nan")

    @staticmethod
    def mean(data_lst, weights=None):
        """Arithmetic mean of list

        Args:
            data_lst (list of floats): List of values to be assessed
            weights (list of floats): Weights for each value
        Returns:
            mean value
        """
        return np.average(data_lst, weights=weights)

    @staticmethod
    def inverse_mean(data_lst, weights=None):
        """Mean of the inverse of each entry

        Args:
            data_lst (list of floats): List of values to be assessed
            weights (list of floats): Weights for each value
        Returns:
            inverse mean
        """
        return PropertyStats.mean([1.0 / x for x in data_lst], weights=weights)

    @staticmethod
    def avg_dev(data_lst, weights=None):
        """Mean absolute deviation of list of element data.

        This is computed by first calculating the mean of the list,
        and then computing the average absolute difference between each value
        and the mean.

        Args:
            data_lst (list of floats): List of values to be assessed
            weights (list of floats): Weights for each value
        Returns:
            mean absolute deviation
        """
        mean = PropertyStats.mean(data_lst, weights)
        return np.average(np.abs(np.subtract(data_lst, mean)), weights=weights)

    @staticmethod
    def std_dev(data_lst, weights=None):
        """Standard deviation of a list of element data

        Args:
            data_lst (list of floats): List of values to be assessed
            weights (list of floats): Weights for each value
        Returns:
            standard deviation
        """
        # Special case: Only one entry
        if len(data_lst) == 1:
            # This prevents numerical issues in the weighted std_dev
            return 0

        if weights is None:
            return np.std(data_lst)
        else:
            beta = np.sum(weights) / (np.sum(weights) ** 2 - np.sum(np.power(weights, 2)))
            dev = np.power(np.subtract(data_lst, PropertyStats.mean(data_lst, weights=weights)), 2)
            return np.sqrt(beta * np.dot(dev, weights))

    @staticmethod
    def skewness(data_lst, weights=None):
        """Skewness of a list of data

        Args:
            data_lst (list of floats): List of values to be assessed
            weights (list of floats): Weights for each value
        Returns:
            shewness
        """
        # Special case: Only one entry
        if len(data_lst) == 1:
            # This prevents numerical issues in the weighted std_dev
            return 0

        if weights is None:
            return stats.skew(data_lst)
        else:
            # Compute the mean
            mean = PropertyStats.mean(data_lst, weights)

            # Compute the second and 3rd moments of the difference from the mean
            total_weight = np.sum(weights)
            diff = np.subtract(data_lst, mean)
            u3 = np.dot(weights, np.power(diff, 3)) / total_weight
            u2 = np.dot(weights, np.power(diff, 2)) / total_weight
            if np.isclose(u3, 0):
                return 0
            return u3 / u2 ** 1.5

    @staticmethod
    def kurtosis(data_lst, weights=None):
        """Kurtosis of a list of data

        Args:
            data_lst (list of floats): List of values to be assessed
            weights (list of floats): Weights for each value
        Returns:
            kurtosis
        """
        # Special case: Only one entry
        if len(data_lst) == 1:
            # This prevents numerical issues in the weighted std_dev
            return 0

        if weights is None:
            return stats.kurtosis(data_lst, fisher=False)
        else:
            # Compute the mean
            mean = PropertyStats.mean(data_lst, weights)

            # Compute the second and 4th moments of the difference from the mean
            total_weight = np.sum(weights)
            diff_sq = np.power(np.subtract(data_lst, mean), 2)
            u4 = np.dot(weights, np.power(diff_sq, 2))
            u2 = np.dot(weights, diff_sq)
            if np.isclose(u4, 0):
                return 0
            return u4 / u2 ** 2 * total_weight

    @staticmethod
    def geom_std_dev(data_lst, weights=None):
        """
        Geometric standard deviation

        Args:
            data_lst (list of floats): List of values to be assessed
            weights (list of floats): Weights for each value
        Returns:
            geometric standard deviation
        """

        # Make fake weights, if none are provided
        if weights is None:
            weights = np.ones_like(data_lst)

        # Compute the geometric std dev
        mean = PropertyStats.holder_mean(data_lst, weights, 0)
        beta = np.sum(weights) / (np.sum(weights) ** 2 - np.sum(np.power(weights, 2)))
        dev = np.log(np.true_divide(data_lst, mean))
        return np.sqrt(np.exp(beta * np.dot(weights, np.power(dev, 2))))

    @staticmethod
    def mode(data_lst, weights=None):
        """Mode of a list of data.

        If multiple elements occur equally-frequently (or same weight, if
        weights are provided), this function will return the minimum of those
        values.

        Args:
            data_lst (list of floats): List of values to be assessed
            weights (list of floats): Weights for each value
        Returns:
            mode
        """
        if weights is None:
            return stats.mode(data_lst).mode[0]
        else:
            # Find the entry(s) with the largest weight
            data_lst = np.array(data_lst)
            weights = np.array(weights)
            most_freq = np.isclose(weights, weights.max())

            # Return the minimum of the most-frequent entries
            return data_lst[most_freq].min()

    @staticmethod
    def holder_mean(data_lst, weights=None, power=1):
        """
        Get Holder mean
        Args:
            data_lst: (list/array) of values
            weights: (list/array) of weights
            power: (int/float/str) which holder mean to compute
        Returns: Holder mean
        """

        if isinstance(power, string_types):
            power = float(power)

        if weights is None:
            if power == -1:
                return scipy.stats.hmean(data_lst)
            elif power == 0:
                return stats.mstats.gmean(data_lst)
            else:
                return np.power(np.mean(np.power(data_lst, power)), 1.0 / power)
        else:
            # Compute the normalization factor
            alpha = sum(weights)

            if power == -1:
                denom = 0
                for idx, x in enumerate(data_lst):
                    denom += weights[idx]/x

                return sum(weights) / denom

            # If power=0, return geometric mean
            elif power == 0:
                return np.product(np.power(data_lst, np.true_divide(weights, np.sum(weights))))
            else:
                return np.power(np.sum(np.multiply(weights, np.power(data_lst, power))) / alpha, 1.0/power)

    @staticmethod
    def sorted(data_lst):
        """
        Returns the sorted data_lst
        """
        return np.sort(data_lst)

    @staticmethod
    def eigenvalues(data_lst, symm = False, sort = False):
        """
        Return the eigenvalues of a matrix as a numpy array
        Args:
            data_lst: (matrix-like) of values
            symm: whether to assume the matrix is symmetric
            sort: wheter to sort the eigenvalues
        Returns: eigenvalues
        """
        eigs = np.linalg.eigvalsh(data_lst) if symm else np.linalg.eigvals(data_lst)
        if sort:
            eigs.sort()
        return eigs

    @staticmethod
    def flatten(data_lst):
        """Returns a flattened copy of data_lst-as a numpy array
        """
        return np.array(data_lst).flatten()
