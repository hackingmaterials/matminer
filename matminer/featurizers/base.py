from __future__ import division, unicode_literals

import sys
import warnings
import pandas as pd
import numpy as np
from six import string_types
from multiprocessing import Pool, cpu_count
from sklearn.base import TransformerMixin, BaseEstimator


class BaseFeaturizer(BaseEstimator, TransformerMixin):
    """
    Abstract class to calculate features from raw materials input data
    such a compound formula or a pymatgen crystal structure or
    bandstructure object.

    ## Using a BaseFeaturizer Class

    There are multiple ways for running the featurize routines:

        `featurize`: Featurize a single entry
        `featurize_many`: Featurize a list of entries
        `featurize_dataframe`: Compute features for many entries, store results as
            columns in a dataframe

    Some featurizers require first calling the `fit` method before the
    featurization methods can function. Generally, you pass the dataset to
    fit to determine which features a featurizer should compute. For example,
    a featurizer that returns the partial radial distribution function
    may need to know which elements are present in a dataset.

    You can also employ the featurizer as part of a ScikitLearn Pipeline object.
    For these cases, scikit-learn calls the `transform` function of the `BaseFeaturizer`
    which is a less-featured wrapper of `featurize_many`. You would then provide your input
    data as an array to the Pipeline, which would output the featurers as an array.

    Beyond the featurizing capability, BaseFeaturizer also includes methods
    for retrieving proper references for a featurizer. The `citations` function
    returns a list of papers that should be cited. The `implementors` function
    returns a list of people who wrote the featurizer, so that you know
    who to contact with questions.

    ## Implementing a New BaseFeaturizer Class

    These operations must be implemented for each new featurizer:
        `featurize` - Takes a single material as input, returns the features of that material.
        `feature_labels` - Generates a human-meaningful name for each of the features
        `citations` - Returns a list of citations in BibTeX format
        `implementors` - Returns a list of people who contributed writing a paper

    None of these operations should change the state of the featurizer. I.e., running each
    method twice should no produce different results, no class attributes should be changed,
    running one operation should not affect the output of another.

    All options of the featurizer must be set by the `__init__` function. All options must
    be listed as keyword arguments, and the value must be saved as a class attribute with
    the same name (e.g., argument `n` should be stored in `self.n`). These requirements
    are necessary for compatibility with the `get_params` and `set_params` methods
    of `BaseEstimator`.

    Depending on the complexity of your featurizer, it may be worthwhile to implement a
    `from_preset` class method. The `from_preset` method takes the name of a preset and
    returns an instance of the featurizer with some hard-coded set of inputs. The `from_preset`
    option is particularly useful for defining the settings used by papers in the literature.

    Optionally, you can implement the `fit` operation if there are attributes of your featurizer that
    must be set for the featurizer to work. Any variables that are set by fitting should be stored
    as class attributes that end with an underscore. (This follows the pattern used by
    scikit-learn).
    """

    def fit(self, X, y=None):
        """Update the parameters of this featurizer based on available data

        Args:
            X - [list of tuples], training data"""
        pass

    def transform(self, X):
        """Compute features for a list of inputs"""

        return self.featurize_many(X, ignore_errors=True)

    def featurize_dataframe(self, df, col_id, ignore_errors=False,
                            inplace=True, n_jobs=1):
        """
        Compute features for all entries contained in input dataframe.

        Args:
            df (Pandas dataframe): Dataframe containing input data.
            col_id (str or list of str): column label containing objects to
                featurize. Can be multiple labels if the featurize function
                requires multiple inputs.
            ignore_errors (bool): Returns NaN for dataframe rows where
                exceptions are thrown if True. If False, exceptions
                are thrown as normal.
            inplace (bool): Whether to add new columns to input dataframe (df)
            n_jobs (int): Number of parallel processes to execute when
                featurizing the dataframe. If None, automatically determines the
                number of processing cores on the system and sets n_procs to
                this number.

        Returns:
            updated dataframe.
        """

        # If only one column and user provided a string, put it inside a list
        if isinstance(col_id, string_types):
            col_id = [col_id]

        # Generate the feature labels
        labels = self.feature_labels()

        # Check names to avoid overwriting the current columns
        for col in df.columns.values:
            if col in labels:
                raise ValueError('"{}" exists in input dataframe'.format(col))

        # Compute the features
        features = self.featurize_many(df[col_id].values, n_jobs, ignore_errors)

        # Create dataframe with the new features
        res = pd.DataFrame(features, index=df.index, columns=labels)

        # Update the existing dataframe
        if inplace:
            for k in self.feature_labels():
                df[k] = res[k]
            return df
        else:
            return pd.concat([df, res], axis=1)

    def featurize_many(self, entries, n_jobs=1, ignore_errors=False):
        """
        Featurize a list of entries.
        If `featurize` takes multiple inputs, supply inputs as a list of tuples.

        Args:
           entries (list): A list of entries to be featurized.

        Returns:
           (list) features for each entry.
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
            return [self.featurize_wrapper(x) for x in entries]
        else:
            if sys.version_info[0] < 3:
                warnings.warn("Multiprocessing dataframes is not supported in "
                              "matminer for Python 2.x. Multiprocessing has "
                              "been disabled. Please upgrade to Python 3.x to "
                              "enable multiprocessing.")
                return self.featurize_many(entries, n_jobs=1, ignore_errors=ignore_errors)
            with Pool(n_jobs) as p:
                return p.map(self.featurize_wrapper, entries)

    def featurize_wrapper(self, x):
        #TODO: documentation!
        try:
            return self.featurize(*x)
        except:
            if self.__ignore_errors:
                return [float("nan")] * len(self.feature_labels())
            else:
                raise

    def featurize(self, *x):
        """
        Main featurizer function, which has to be implemented
        in any derived featurizer subclass.

        Args:
            x: input data to featurize (type depends on featurizer).

        Returns:
            (list) one or more features.
        """

        raise NotImplementedError("featurize() is not defined!")

    def feature_labels(self):
        """
        Generate attribute names.

        Returns:
            ([str]) attribute labels.
        """

        raise NotImplementedError("feature_labels() is not defined!")

    def citations(self):
        """
        Citation(s) and reference(s) for this feature.

        Returns:
            (list) each element should be a string citation,
                ideally in BibTeX format.
        """

        raise NotImplementedError("citations() is not defined!")

    def implementors(self):
        """
        List of implementors of the feature.

        Returns:
            (list) each element should either be a string with author name (e.g.,
                "Anubhav Jain") or a dictionary  with required key "name" and other
                keys like "email" or "institution" (e.g., {"name": "Anubhav
                Jain", "email": "ajain@lbl.gov", "institution": "LBNL"}).
        """

        raise NotImplementedError("implementors() is not defined!")


class MultipleFeaturizer(BaseFeaturizer):
    """
    Class that runs multiple featurizers on the same data
    All featurizers must take the same kind of data as input
    to the featurize function."""

    def __init__(self, featurizers):
        """
        Create a new instance of this featurizer.

        Args:
            featurizers ([BaseFeaturizer]): list of featurizers to run.
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
