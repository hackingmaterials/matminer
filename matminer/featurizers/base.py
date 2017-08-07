import numpy as np

class BaseFeaturizer(object):
    """Abstract class to calculate attributes for compounds"""

    def featurize_dataframe(self, df, col_id):
        """
        Compute features for all entries contained in input dataframe
        
        Args: 
            df (Pandas dataframe): Dataframe containing input data
            col_id (string): column label containing objects to featurize

        Returns:
            updated Dataframe
        """

        features = []
        x_list = df[col_id]
        for x in x_list:
            features.append(self.featurize(x))
        
        features = np.array(features)

        labels = self.feature_labels()
        df = df.assign(**dict(zip(labels, features.T)))
        return df
    
    def featurize(self, x):
        """
        Main featurizer function. Only defined in feature subclasses.

        Args:
            x: input data to featurize (type depends on featurizer)

        Returns:
            list of one or more features
        """

        raise NotImplementedError("featurize() is not defined!")
    
    def feature_labels(self):
        """
        Generate attribute names
        
        Returns:
            list of strings for attribute labels
        """

        raise NotImplementedError("feature_labels() is not defined!")

    def citations(self):
        """
        Citation / reference for feature

        Returns:
            array - each element should be str citation, ideally in BibTeX format
        """

        raise NotImplementedError("citations() is not defined!")

    def implementors(self):
        """
        List of implementors of the feature.

        Returns:
            array - each element should either be str with author name (e.g., "Anubhav Jain") or
                dict with required key "name" and other keys like "email" or "institution" (e.g.,
                {"name": "Anubhav Jain", "email": "ajain@lbl.gov", "institution": "LBNL"}).
        """

        raise NotImplementedError("implementors() is not defined!")
