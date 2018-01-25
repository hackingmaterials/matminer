from __future__ import division

import pandas as pd
import numpy as np
from six import string_types
from multiprocessing import Pool, cpu_count
from functools import partial


def featurize_wrapper(x_split, ignore_errors, labels, f_obj):
    """

    A multiprocessing-oriented wrapper for BaseFeaturizer().featurize

    Args:
        x_split (ndarray): Array of x vectors to use for featurization
            by this process
        ignore_errors (bool): Returns NaN for dataframe rows where the
            exceptions are thrown if True. If False, exceptions are
            thrown as normal.
        labels: The set of featurization labels for the f_obj object.
        f_obj (BaseFeaturizer): The featurizer instance undergoing
            featurization.
    Returns:
        features (list): The list of features computed based on x_list
    """

    features = []
    for x in x_split.values:
        try:
            features.append(f_obj.featurize(*x))
        except:
            if ignore_errors:
                features.append([float("nan")] * labels)
            else:
                raise
    return features


class BaseFeaturizer(object):
    """Abstract class to calculate attributes for compounds"""

    def featurize_dataframe(self, df, col_id, ignore_errors=False,
                            inplace=True, multiindex=False, n_procs=1):
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
            multiindex (bool): Whether multiple levels of column header will
                show up in the table. When True, column headings are grouped
                and labelled by their featurizer class.
            n_procs (int, str): Number of parallel processes to execute when
                featurizing the dataframe. 'auto' automatically determines the
                number of processing cores on the system and sets n_procs to
                this number.
        Returns:
            updated Dataframe
        """

        # If only one column and user provided a string, put it inside a list
        if isinstance(col_id, string_types):
            col_id = [col_id]

        # Generate the feature labels
        labels = self.feature_labels()
        if multiindex:
            cols = pd.MultiIndex.from_product(
                [[self.__class__.__name__], labels])
        else:
            cols = labels

        # Compute the features
        x_list = df[col_id]
        n_procs = cpu_count() if n_procs == 'auto' else n_procs
        pool = Pool(n_procs)
        x_split = np.array_split(x_list, n_procs)
        featurize = partial(featurize_wrapper,
                            ignore_errors=ignore_errors,
                            labels=labels,
                            f_obj=self)
        features = [i for j in pool.map(featurize, x_split) for i in j]
        pool.close()
        pool.join()

        # Create dataframe with the new features
        res_df = pd.DataFrame(features, index=df.index, columns=cols)

        # Update the existing dataframe
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
