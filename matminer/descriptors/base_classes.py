from pymatgen import Composition
import numpy as np
import pandas as pd

class BaseFeaturizer(object):
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
        comp_frame = comp_frame.assign(**dict(zip(labels, features.T)))
        return comp_frame
    
    def featurize(self, comp_obj):
        """
        Main featurizer function. Only defined in feature subclasses.

        Args:
            comp_obj: Pymatgen composition object

        Returns:
            list of features
        """

        raise NotImplementedError("Featurizer is not defined")
    
    def generate_labels(self):
        """
        Generate attribute names
        
        Returns:
            list of strings for attribute labels
        """

        raise NotImplementedError("Featurizer is not defined")


