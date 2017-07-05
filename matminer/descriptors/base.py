from __future__ import division, unicode_literals, print_function

import abc
import six

import numpy as np

from pymatgen import Composition


class BaseFeaturizer(six.with_metaclass(abc.ABCMeta)):
    """Abstract class to calculate attributes for compounds"""

    def featurize_all(self, comp_frame, col_id="composition"):
        """
        Compute features for all compounds contained in input dataframe
        
        Args: 
            comp_frame (Pandas dataframe): Dataframe containing column of compounds
            col_id (string): column label containing compositions

        Returns:
            updated Dataframe
        """

        features = []
        comp_list = comp_frame[col_id]
        for comp in comp_list:
            comp_obj = Composition(comp)
            features.append(self.featurize(comp_obj))
        
        features = np.array(features)

        labels = self.generate_labels()
        comp_frame = comp_frame.assign(**dict(zip(labels, [features[:,i] for i in range(np.shape(features)[1])])))

        return comp_frame

    @abc.abstractmethod
    def featurize(self, comp_obj):
        """
        Main featurizer function. Only defined in feature subclasses.

        Args:
            comp_obj: Pymatgen composition object

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
