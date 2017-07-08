from __future__ import division, unicode_literals, print_function

"""
Defines abstract featurizer class. Inspired by deepchem model.
"""

import abc
import six


class AbstractFeaturizer(six.with_metaclass(abc.ABCMeta)):
    """
    Abstract class to calculate attributes
    """

    @abc.abstractmethod
    def featurize(self):
        """
        Main featurizer function. Only defined in feature subclasses.

        Returns:
            list of features
        """
        pass

    @abc.abstractmethod
    def generate_labels(self):
        """
        Generate attribute names
        
        Returns:
            list of strings for attribute labels
        """
        pass
