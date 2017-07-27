import numpy as np
import pandas as pd

class BaseFeaturizer(object):
    """Abstract class to calculate attributes for compounds"""

    def featurize_dataframe(self, comp_frame, col_id="composition"):
        """
        Compute features for all compounds contained in input dataframe
        
        Args: 
            comp_frame (Pandas dataframe): Dataframe containing column of compounds
            col_id (string): column label containing objects to featurize

        Returns:
            updated Dataframe
        """

        features = []
        comp_list = comp_frame[col_id]
        for comp in comp_list:
            features.append(self.featurize(comp))
        
        features = np.array(features)

        labels = self.feature_labels()
        comp_frame = comp_frame.assign(**dict(zip(labels, features.T)))
        return comp_frame
    
    def featurize(self, comp):
        """
        Main featurizer function. Only defined in feature subclasses.

        Args:
            comp: Pymatgen composition object

        Returns:
            list of features
        """

        raise NotImplementedError("Featurizer is not defined")
    
    def feature_labels(self):
        """
        Generate attribute names
        
        Returns:
            list of strings for attribute labels
        """

        raise NotImplementedError("Featurizer is not defined")

    def credits(self):
        """
        Citation for feature

        Returns:
            BibTeX citation
        """

        raise NotImplementedError("Featurizer is not defined")
