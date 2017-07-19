"""
File containing general methods for computing property statistics
"""
import numpy as np
from monty.design_patterns import singleton

@singleton
class PropertyStats(object):

    def __init__(self):
        self.stat_dict = {"minimum": self.minimum, "maximum":self.maximum, "range":self.attr_range, "mean":self.mean,
            "avg_dev":self.avg_dev, "std_dev":self.std_dev, "mode":self.mode}

    def minimum(self, elem_data):
        """
        Minimum value in a list of element data
        Args:
            elem_data (list of floats): Value of a property for each atom in a compound
        Returns: 
            minimum value"""
        return min(elem_data)

    def maximum(self, elem_data):
        """
        Maximum value in a list of element data
        Args:
            elem_data (list of floats): Value of a property for each atom in a compound
        Returns: 
            maximum value"""
        return max(elem_data)

    def attr_range(self, elem_data):
        """
        Range of a list of element data
        Args:
            elem_data (list of floats): Value of a property for each atom in a compound
        Returns: 
            range"""
        return max(elem_data) - min(elem_data)

    def mean(self, elem_data):
        """
        Mean of list of element data
        Args:
            elem_data (list of floats): Value of a property for each atom in a compound
        Returns: 
            mean value"""
        return sum(elem_data)/len(elem_data)

    def avg_dev(self, elem_data):
        """
        Average absolute deviation of list of element data
        Args:
            elem_data (list of floats): Value of a property for each atom in a compound
        Returns: 
            average absolute deviation"""
        return sum(np.abs(np.subtract(elem_data, self.mean(elem_data))))/len(elem_data)

    def std_dev(self, elem_data):
        """
        Standard deviation of a list of element data
        Args:
            elem_data (list of floats): Value of a property for each atom in a compound
        """
        return np.std(elem_data)

    def mode(self, elem_data):
        """
        Mode of a list of element data
        Args:
            elem_data (list of floats): Value of a property for each atom in a compound
        Returns: 
            mode"""
        return max(set(elem_data), key=elem_data.count)

    def calc_stat(self, elem_data, stat):
        """
        Compute a property statistic
        """
        return self.stat_dict[stat](elem_data)
