from __future__ import division, unicode_literals

import sys
import warnings
import pandas as pd
import numpy as np
from six import string_types
from multiprocessing import Pool, cpu_count
from functools import partial


class BaseFeaturizer(object):
    """Abstract class to calculate attributes for compounds"""

    def featurize_dataframe(self, df, col_id, ignore_errors=False,
                            inplace=True, label_props=None, n_jobs=1,
                            **kwargs):
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
            n_jobs (int): Number of parallel processes to execute when
                featurizing the dataframe. If None, automatically determines the
                number of processing cores on the system and sets n_procs to
                this number.
            label_props (dict): properties to be fed as kwargs to
                feature_labels, e. g. {"postprocess": sympy.latex}
            **kwargs (kwargs): kwargs to be passed to featurize()
        Returns:
            updated Dataframe
        """

        # If only one column and user provided a string, put it inside a list
        if isinstance(col_id, string_types):
            col_id = [col_id]

        # Generate the feature labels
        label_props = label_props if label_props is not None else {}
        labels = self.feature_labels(**label_props)

        # Compute the features
        features = self.featurize_many(df[col_id].values, n_jobs, ignore_errors,
                                       **kwargs)

        # Create dataframe with the new features

        res = pd.DataFrame(features, index=df.index, columns=labels)

        # Update the existing dataframe
        if inplace:
            for k in self.feature_labels():
                df[k] = res[k]
            return df
        else:
            return pd.concat([df, res], axis=1)

    def featurize_many(self, entries, n_jobs=1, ignore_errors=False, **kwargs):
        """
        Featurize a list of entries.

        If `featurize` takes multiple inputs, supply inputs as a list of tuples.

        Args:
           entries (list): A list of entries to be featurized
        Returns:
           list - features for each entry
        """

        self.__ignore_errors = ignore_errors

        # Check inputs
        if not hasattr(entries, '__getitem__'):
            raise Exception("'entries' must be a list-like object")

        # Special case: Empty list
        if len(entries) is 0:
            return []

        # If the featurize function only has a single arg, zip the inputs
        if not isinstance(entries[0], (tuple, list, np.ndarray)):
            entries = zip(entries)

        # set the number of processes to the number of cores on the system
        n_jobs = cpu_count() if n_jobs is None else n_jobs

        # Run the actual featurization
        if n_jobs == 1:
            return [self.featurize_wrapper(x, **kwargs) for x in entries]
        else:
            if sys.version_info[0] < 3:
                warnings.warn("Multiprocessing dataframes is not supported in "
                              "matminer for Python 2.x. Multiprocessing has "
                              "been disabled. Please upgrade to Python 3.x to "
                              "enable multiprocessing.")
                return self.featurize_many(entries, n_jobs=1,
                                           ignore_errors=ignore_errors,
                                           **kwargs)
            with Pool(n_jobs) as p:
                pfunc = partial(self.featurize_wrapper, **kwargs)
                return p.map(pfunc, entries)

    def featurize_wrapper(self, x, **kwargs):
        try:
            return self.featurize(*x, **kwargs)
        except:
            if self.__ignore_errors:
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

    def feature_labels(self, col_id=None):
        """
        Generate attribute names

        Args:
            col_id: some featurizers (e. g. Functional Featurizer)
                need column information to generate labels

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
