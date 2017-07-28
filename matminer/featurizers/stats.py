"""
File containing general methods for computing property statistics
"""
import numpy as np
from scipy import stats

from six import string_types


class PropertyStats(object):

    @staticmethod
    def calc_stat(stat, data_lst, weights=None):
        """
        Compute a property statistic

        Args:
            str (str) - Name of property to be compute. If there are arguments to the statistics function, these
             should be added after the name and separated by two underscores. For example, the 2nd Holder mean would
             be "holder_mean__2"
            data_lst (list of floats): list of values
            weights (list of floats): (Optional) weights for each element in data_lst
        Reteurn:
            float - Desired statistic
        """
        statistics = stat.split("__")
        return getattr(PropertyStats, statistics[0])(data_lst, weights, *statistics[1:])

    @staticmethod
    def minimum(data_lst, weights=None):
        """
        Minimum value in a list of element data
        Args:
            data_lst (list of floats): Value of a property for each atom in a compound
            weights (ignored)
        Returns: 
            minimum value"""
        return min(data_lst)

    @staticmethod
    def maximum(data_lst, weights=None):
        """
        Maximum value in a list of element data
        Args:
            data_lst (list of floats): Value of a property for each atom in a compound
            weights (ignored)
        Returns: 
            maximum value"""
        return max(data_lst)

    @staticmethod
    def range(data_lst, weights=None):
        """
        Range of a list of element data
        Args:
            data_lst (list of floats): Value of a property for each atom in a compound
            weights (ignored)
        Returns: 
            range"""
        return max(data_lst) - min(data_lst)

    @staticmethod
    def mean(data_lst, weights=None, **kwargs):
        """
        Mean of list of element data
        Args:
            data_lst (list of floats): Value of a property for each atom or element in a compound
            weights (list of floats): Weights for each value
        Returns: 
            mean value"""
        if weights is None:
            return np.average(data_lst)
        else:
            return np.dot(data_lst, weights) / sum(weights)

    @staticmethod
    def avg_dev(data_lst, weights=None):
        """
        Average absolute deviation of list of element data
        Args:
            data_lst (list of floats): Value of a property for each atom in a compound
            weights (list of floats): Atomic fractions
        Returns: 
            average absolute deviation"""
        mean = PropertyStats.mean(data_lst, weights)
        return np.average(np.abs(np.subtract(data_lst, mean)), weights=weights)

    @staticmethod
    def std_dev(data_lst, weights=None):
        """
        Standard deviation of a list of element data
        Args:
            data_lst (list of floats): Value of a property for each atom in a compound
            weights (list of floats): Atomic fractions
        """
        if weights is None:
            return np.std(data_lst)
        else:
            dev = np.subtract(data_lst, PropertyStats.mean(data_lst, weights=weights))**2
            return np.sqrt(PropertyStats.mean(dev, weights=weights))

    @staticmethod
    def mode(data_lst, weights=None):
        """
        Mode of a list of element data. If multiple elements occur equally-frequently (or same weight, if weights are
        provided), this function will return the average of those values
        Args:
            data_lst (list of floats): Value of a property for each atom in a compound
        Returns: 
            mode"""
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
            if power == 0:
                return stats.mstats.gmean(data_lst)
            else:
                return np.power(np.mean(np.power(data_lst, power)), 1.0 / power)
        else:
            # Compute the normalization factor
            alpha = sum(weights)

            # If power=0, return geometric mean
            if power == 0:
                return np.product(np.power(data_lst, np.divide(weights, np.sum(weights))))
            else:
                return np.power(np.sum(np.multiply(weights, np.power(data_lst, power))) / alpha, 1.0/power)

