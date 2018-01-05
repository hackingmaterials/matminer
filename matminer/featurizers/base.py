from __future__ import division

import pandas as pd
from six import string_types


class BaseFeaturizer(object):
    """Abstract class to calculate attributes for compounds"""

    def featurize_dataframe(self, df, col_id, ignore_errors=False, inplace=True):
        """
        Compute features for all entries contained in input dataframe
        
        Args: 
            df (Pandas dataframe): Dataframe containing input data
            col_id (str or list of str): column label containing objects to
                featurize. Can be multiple labels if the featurize function
                requires multiple inputs
            ignore_errors (bool): Returns NaN for dataframe rows where
                exceptions are thrown if True. If False, exceptions
                are thrown as normal.
            inplace (bool): Whether to add new columns to input dataframe (df)

        Returns:
            updated Dataframe
        """

        # If only one column and user provided a string, put it inside a list
        if isinstance(col_id, string_types):
            col_id = [col_id]

        # Compute the features
        features = []
        x_list = df[col_id]
        for x in x_list.values:
            try:
                features.append(self.featurize(*x))
            except:
                if ignore_errors:
                    features.append([float("nan")]
                                    * len(self.feature_labels()))
                else:
                    raise

        # Generate the feature labels
        labels = self.feature_labels()

        # Create dataframe with the new features
        new_cols = dict(zip(labels, [pd.Series(x) for x in zip(*features)]))

        # Update the dataframe
        if inplace:
            for key, value in new_cols.items():
                df[key] = value
            return df
        else:
            return df.assign(**new_cols)

    def featurize(self, *x):
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
            array - each element should be str citation, ideally in BibTeX
                format
        """

        raise NotImplementedError("citations() is not defined!")

    def implementors(self):
        """
        List of implementors of the feature

        Returns:
            array - each element should either be str with author name (e.g.,
                "Anubhav Jain") or dict with required key "name" and other
                keys like "email" or "institution" (e.g., {"name": "Anubhav
                Jain", "email": "ajain@lbl.gov", "institution": "LBNL"}).
        """

        raise NotImplementedError("implementors() is not defined!")
