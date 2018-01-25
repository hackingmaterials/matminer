from __future__ import division

import pandas as pd
import numpy as np
from six import string_types


class BaseFeaturizer(object):
    """Abstract class to calculate attributes for compounds"""

    def featurize_dataframe(self, df, col_id, ignore_errors=False, inplace=True,
                            multiindex=False):
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
            multiindex (bool): Whether to create tiered columns

        Returns:
            updated Dataframe
        """

        # If only one column and user provided a string, put it inside a list
        if isinstance(col_id, string_types):
            col_id = [col_id]

        # Define all arguments to the featurizer in Series of tuples, then featurize
        args = pd.Series.from_array(zip(*[df[cid] for cid in col_id]))
        feature_vals = args.apply(lambda x: self.featurize_wrapper(ignore_errors, *x))

        if multiindex:
            cols = pd.MultiIndex.from_product([[self.__class__.__name__],
                                               self.feature_labels()])
        else:
            cols = self.feature_labels()
        res_df = pd.DataFrame(feature_vals.values.tolist(), index=df.index, columns=cols)

        if inplace:
            for k in self.feature_labels():
                df[k] = res_df[k]
            return df
        else:
            if multiindex and df.columns.nlevels < 2:
                # Add an 'original data' multiindex to the input df
                df.columns = pd.MultiIndex.from_product([["Original Data"],
                                                         df.columns.values])
            return pd.concat([df, res_df], axis=1)


    def featurize_wrapper(self, ignore_errors, *x):
        """
        Error wrapper for featurizer.

        Args:
            ignore_errors (bool): features return nan arrays if True
            x: input data to featurizer

        Returns:
            list of one or more features from featurize
        """
        try:
            return self.featurize(*x)
        except:
            if ignore_errors:
                return [float("nan")] * len(self.feature_labels())
            else:
                raise


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


class MultipleFeaturizer(BaseFeaturizer):
    """Class that runs multiple featurizers on the same data

    All featurizers must take the same kind of data as input to the featurize function."""

    def __init__(self, featurizers):
        """Create a new instance of this featurizer

        Args:
            featurizers - [BaseFeaturizer], list of featurizers to run
        """
        self.featurizers = featurizers

    def featurize(self, *x):
        return np.hstack(f.featurize(*x) for f in self.featurizers)

    def feature_labels(self):
        return sum([f.feature_labels() for f in self.featurizers], [])

    def citations(self):
        return list(set(sum([f.citations() for f in self.featurizers], [])))

    def implementors(self):
        return list(set(sum([f.implementors() for f in self.featurizers], [])))
