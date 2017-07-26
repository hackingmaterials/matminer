"""
File containing general methods for computing property statistics
"""
import numpy as np
from monty.design_patterns import singleton

@singleton
class PropertyStats(object):

    def __init__(self):
        self.stat_dict = {"minimum": self.minimum, "maximum":self.maximum, "range":self.attr_range, "mean":self.mean,
            "avg_dev":self.avg_dev, "std_dev":self.std_dev, "mode":self.mode, "holder_mean":self.holder_mean}

    def minimum(self, data_lst, **kwargs):
        """
        Minimum value in a list of element data
        Args:
            data_lst (list of floats): Value of a property for each atom in a compound
            weights (ignored)
        Returns: 
            minimum value"""
        return min(data_lst)

    def maximum(self, data_lst, **kwargs):
        """
        Maximum value in a list of element data
        Args:
            data_lst (list of floats): Value of a property for each atom in a compound
            weights (ignored)
        Returns: 
            maximum value"""
        return max(data_lst)

    def attr_range(self, data_lst, **kwargs):
        """
        Range of a list of element data
        Args:
            data_lst (list of floats): Value of a property for each atom in a compound
            weights (ignored)
        Returns: 
            range"""
        return max(data_lst) - min(data_lst)

    def mean(self, data_lst, weights=None, **kwargs):
        """
        Mean of list of element data
        Args:
            data_lst (list of floats): Value of a property for each atom or element in a compound
            weights (list of floats): Atomic fractions
        Returns: 
            mean value"""
        if weights is None:
            return sum(data_lst)/len(data_lst)
        else:
            return sum([data_lst[i]*weights[i] for i in range(len(data_lst))])

    def avg_dev(self, data_lst, weights=None, **kwargs):
        """
        Average absolute deviation of list of element data
        Args:
            data_lst (list of floats): Value of a property for each atom in a compound
            weights (list of floats): Atomic fractions
        Returns: 
            average absolute deviation"""
        if weights is None:
            return sum(abs(np.subtract(data_lst, self.mean(data_lst))))/len(data_lst)
        else:
            abs_dev = abs(np.subtract(data_lst, self.mean(data_lst, weights=weights)))
            return self.mean(abs_dev, weights=weights)

    def std_dev(self, data_lst, weights=None, **kwargs):
        """
        Standard deviation of a list of element data
        Args:
            data_lst (list of floats): Value of a property for each atom in a compound
            weights (list of floats): Atomic fractions
        """
        if weights is None:
            return np.std(data_lst)
        else:
            dev = np.subtract(data_lst, self.mean(data_lst, weights=weights))**2
            return np.sqrt(self.mean(dev, weights=weights))

    def mode(self, data_lst, weights=None, **kwargs):
        """
        Mode of a list of element data
        Args:
            data_lst (list of floats): Value of a property for each atom in a compound
        Returns: 
            mode"""
        if weights is None:
            return max(set(data_lst), key=data_lst.count)
        else:
            ind_max = max(xrange(len(weights)), key=weights.__getitem__)
            return data_lst[ind_max]

    def holder_mean(data_lst, power=None, **kwargs):
        """
        Get Holder mean
        Args:
            data_lst: (list/array) of values
            power: (int/float) non-zero real number
        Returns: Holder mean
        """
        # Function for calculating Geometric mean
        geomean = lambda n: reduce(lambda x, y: x * y, n) ** (1.0 / len(n))

        # If power=0, return geometric mean
        if power == 0:
            return geomean(data_lst)

        else:
            total = 0.0
            for value in data_lst:
                total += value ** power
            return (total / len(data_lst)) ** (1 / float(power))

    def calc_stat(self, data_lst, stat, **kwargs):
        """
        Compute a property statistic
        """
        return self.stat_dict[stat](data_lst, **kwargs)
