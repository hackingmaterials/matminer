from __future__ import division, unicode_literals

import sys
import traceback
import warnings
from abc import ABC, abstractmethod
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from six import reraise, string_types
from sklearn.base import BaseEstimator, TransformerMixin, is_classifier
from tqdm.auto import tqdm

from matminer.utils.utils import homogenize_multiindex


class BaseFeaturizer(BaseEstimator, TransformerMixin, ABC):
    """
    Abstract class to calculate features from raw materials input data
    such a compound formula or a pymatgen crystal structure or
    bandstructure object.

    ## Using a BaseFeaturizer Class

    There are multiple ways for running the featurize routines:

        `featurize`: Featurize a single entry
        `featurize_many`: Featurize a list of entries
        `featurize_dataframe`: Compute features for many entries, store results
            as columns in a dataframe

    Some featurizers require first calling the `fit` method before the
    featurization methods can function. Generally, you pass the dataset to
    fit to determine which features a featurizer should compute. For example,
    a featurizer that returns the partial radial distribution function
    may need to know which elements are present in a dataset.

    You can can also use the `precheck` and `precheck_dataframe` methods to
    ensure a featurizer is in scope for a given sample (or dataset) before
    featurizing.

    You can also employ the featurizer as part of a ScikitLearn Pipeline object.
    For these cases, ScikitLearn calls the `transform` function of the
    `BaseFeaturizer` which is a less-featured wrapper of `featurize_many`. You
    would then provide your input data as an array to the Pipeline, which would
    output the features as an array.

    Beyond the featurizing capability, BaseFeaturizer also includes methods
    for retrieving proper references for a featurizer. The `citations` function
    returns a list of papers that should be cited. The `implementors` function
    returns a list of people who wrote the featurizer, so that you know
    who to contact with questions.

    ## Implementing a New BaseFeaturizer Class

    These operations must be implemented for each new featurizer:
        `featurize` - Takes a single material as input, returns the features of
            that material.
        `feature_labels` - Generates a human-meaningful name for each of the
            features.
        `citations` - Returns a list of citations in BibTeX format
        `implementors` - Returns a list of people who contributed to writing a
            paper.

    None of these operations should change the state of the featurizer. I.e.,
    running each method twice should not produce different results, no class
    attributes should be changed, and running one operation should not affect
    the output of another.

    All options of the featurizer must be set by the `__init__` function. All
    options must be listed as keyword arguments with default values, and the
    value must be saved as a class attribute with the same name (e.g., argument
    `n` should be stored in `self.n`). These requirements are necessary for
    compatibility with the `get_params` and `set_params` methods of
    `BaseEstimator`, which enable easy interoperability with ScikitLearn

    Depending on the complexity of your featurizer, it may be worthwhile to
    implement a `from_preset` class method. The `from_preset` method takes the
    name of a preset and returns an instance of the featurizer with some
    hard-coded set of inputs. The `from_preset` option is particularly useful
    for defining the settings used by papers in the literature.

    Optionally, you can implement the `fit` operation if there are attributes of
    your featurizer that must be set for the featurizer to work. Any variables
    that are set by fitting should be stored as class attributes that end with
    an underscore. (This follows the pattern used by ScikitLearn).

    Another option to consider is whether it is worth making any utility
    operations for your featurizer. `featurize` must return a list of features,
    but this may not be the most natural representation for your features (e.g.,
    a `dict` could be better). Making a separate function for computing features
    in this natural representation and having the `featurize` function call this
    method and then convert the data into a list is a recommended approach.
    Users who want to compute the representation in the natural form can use the
    utility function and users who want the data in a ML-ready format (list) can
    call `featurize`. See `PartialRadialDistributionFunction` for an example of
    this concept.

    An additional factor to consider is the chunksize for data parallelisation.
    For lightweight computational tasks, the overhead associated with passing
    data from `multiprocessing.Pool.map()` to the function being parallelized
    can increase the time taken for all tasks to be completed. By setting
    the `self._chunksize` argument, the overhead associated with passing data
    to the tasks can be reduced. Note that there is only an advantage to using
    chunksize when the time taken to pass the data from `map` to the function
    call is within several orders of magnitude to that of the function call
    itself. By default, we allow the Python multiprocessing library to determine
    the chunk size automatically based on the size of the list being featurized.
    You may want to specify a small chunk size for computationally-expensive
    featurizers, which will enable better distribution of tasks across threads.
    In contrast, for more lightweight featurizers, it is recommended that
    the implementor trial a range of chunksize values to find the optimum.
    As a general rule of thumb, if the featurize function takes 0.1 seconds or
    less, a chunksize of around 30 will perform best.

    ## Documenting a BaseFeaturizer

    The class documentation for each featurizer must contain a description of
    the options and the features that will be computed. The options of the class
    must all be defined in the `__init__` function of the class, and we
    recommend documenting them using the
    [Google style](https://google.github.io/styleguide/pyguide.html).

    For auto-generated documentation purposes, the first line of the featurizer
    doc should come under the class declaration (not under __init__) and should
    be a one line summary of the featurizer.

    We recommend starting the class documentation with a high-level overview of
    the features. For example, mention what kind of characteristics of the
    material they describe and refer the reader to a paper that describes these
    features well (use a hyperlink if possible, so that the readthedocs will
    link to that paper). Then, describe each of the individual features in a
    block named "Features". It is necessary here to give the user enough
    information for user to map a feature name what it means. The objective in
    this part is to allow people to understand what each column of their
    dataframe is without having to read the Python code. You do not need to
    explain all of the math/algorithms behind each feature for them to be able
    to reproduce the feature, just to get an idea what it is.
    """

    def set_n_jobs(self, n_jobs):
        """Set the number of threads for this."""
        self._n_jobs = n_jobs

    @property
    def n_jobs(self):
        return self._n_jobs if hasattr(self, '_n_jobs') else cpu_count()

    def set_chunksize(self, chunksize):
        """Set the chunksize used for Pool.map parallelisation."""
        self._chunksize = chunksize

    @property
    def chunksize(self):
        return self._chunksize if hasattr(self, '_chunksize') else None

    def precheck_dataframe(self, df, col_id, return_frac=True, inplace=False) \
            -> [float, pd.DataFrame]:
        """
        Precheck an entire dataframe. Subclasses wanting to use precheck
        functionality should not override this method, they should override
        precheck (unless the entire df determines whether single entries pass
        or fail a precheck).

        Prechecking should be a quick and useful way to check that for a
        particular dataframe (set of featurizer inputs), the featurizer is:

            1. in scope, and/or...
            2. robust to errors and/or...
            3. any other reason you would not practically want to use this
                featurizer in on this dataframe.

        By prechecking before featurizing, you can avoid applying featurizers
        to data that will ultimately fail, return unreliable numbers, or
        are out of scope. Prechecking is also a good time to throw/observe
        warnings (such as long runtime warnings!).

        Args:
            df (pd.DataFrame): A dataframe
            col_id (str or [str]): column label containing objects to featurize.
                Can be multiple labels if the featurize function requires
                multiple inputs.
            return_frac (bool): If True, returns the fraction of entries
                passing the precheck (e.g., 0.5). Else, returns a dataframe.
            inplace (bool); Only relevant if return_frac=False. If inplace=True,
                the input dataframe is modified in memory with a boolean column
                for precheck. Otherwise, a new df with this column is returned.

        Returns:
            (bool, pd.DataFrame): If return_frac=True, returns the fraction of
                entries passing the precheck. Else, returns the dataframe with
                an extra boolean column added for the precheck.

        """
        col_id = [col_id] if isinstance(col_id, string_types) else col_id
        prechecks = [self.precheck(*entries) for entries in df[col_id].values]

        if return_frac:
            return np.sum(prechecks) / len(prechecks)
        else:
            precheck_col = "{} precheck pass".format(self.__class__.__name__)
            if inplace:
                df[precheck_col] = prechecks
            else:
                res = pd.DataFrame({precheck_col: prechecks})
                df = pd.concat([df, res], axis=1)
            return df

    def precheck(self, *x) -> bool:
        """
        Precheck (provide an estimate of whether a featurizer will work or not)
        for a single entry (e.g., a single composition). If the entry fails the
        precheck, it will most likely fail featurization; if it passes, it is
        likely (but not guaranteed) to featurize correctly.

        Prechecks should be:
            * accurate (but can be good estimates rather than ground truth)
            * fast to evaluate
            * unlikely to be obsolete via changes in the featurizer in the near
                future

        This method should be overridden by any featurizer requiring its
        use, as by default all entries will pass prechecking. Also, precheck
        is a good opportunity to throw warnings about long runtimes (e.g., doing
        nearest neighbors computations on a structure with many thousand sites).

        See the documentation for precheck_dataframe for more information.

        Args:
            *x (Composition, Structure, etc.): Input to-be-featurized. Can be
                a single input or multiple inputs.

        Returns:
            (bool): True, if passes the precheck. False, if fails.

        """
        return True

    def fit(self, X, y=None, **fit_kwargs):
        """Update the parameters of this featurizer based on available data

        Args:
            X - [list of tuples], training data
        Returns:
            self
            """
        return self

    def transform(self, X):
        """Compute features for a list of inputs"""

        return self.featurize_many(X, ignore_errors=True, pbar=False)

    def fit_featurize_dataframe(self, df, col_id, fit_args=None,
                                *args, **kwargs):
        """
        The dataframe equivalent of fit_transform. Takes a dataframe and
        column id as input, fits the featurizer to that dataframe, and
        returns a featurized dataframe. Accepts the same arguments as
        featurize_dataframe.

        Args:
            df (Pandas dataframe): Dataframe containing input data.
            col_id (str or list of str): column label containing objects to
                featurize. Can be multiple labels if the featurize function
                requires multiple inputs.
            fit_args (list): list of arguments for fit function.

        Returns:
            updated dataframe based on featurizer fitted to that dataframe.
        """
        if fit_args is None:
            fit_args = []
        return self.fit(df[col_id], *fit_args).featurize_dataframe(df, col_id,
                                                                   *args,
                                                                   **kwargs)

    def featurize_dataframe(self, df, col_id, ignore_errors=False,
                            return_errors=False, inplace=False,
                            multiindex=False, pbar=True):
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
            return_errors (bool). Returns the errors encountered for each
                row in a separate `XFeaturizer errors` column if True. Requires
                ignore_errors to be True.
            inplace (bool): If True, adds columns to the original object in
                memory and returns None. Else, returns the updated object.
                Should be identical to pandas inplace behavior.
            multiindex (bool): If True, use a Featurizer - Feature 2-level
                index using the MultiIndex capabilities of pandas. If done
                inplace, multiindex featurization will overwrite the original
                dataframe's column index.
            pbar (bool): Shows a progress bar if True.

        Returns:
            updated dataframe.
        """

        # If only one column and user provided a string, put it inside a list
        col_id = [col_id] if isinstance(col_id, string_types) else col_id

        # Multiindexing doesn't play nice with other options!
        if multiindex:
            if inplace:
                warnings.warn("Multiindexing enabled with inplace=True! The "
                              "original dataframe index has changed.")

        elif isinstance(df.columns, pd.MultiIndex):
            # If input df is multi, but multi not enabled...
            raise ValueError("Please enable multiindexing to featurize an input"
                             " dataframe containing a column multiindex.")

        # Generate the labels for the columns
        labels = self._generate_column_labels(multiindex, return_errors)

        # Check names to avoid overwriting the current columns
        # ConversionFeaturizer have attribute called _overwrite_data which
        # determines whether an Error is thrown
        overwrite = getattr(self, '_overwrite_data', False)
        if not overwrite:
            for col in df.columns.values:
                if col in labels:
                    raise ValueError(
                        '"{}" exists in input dataframe'.format(col))

        # Compute the features
        features = self.featurize_many(df[col_id].values,
                                       ignore_errors=ignore_errors,
                                       return_errors=return_errors,
                                       pbar=pbar)

        # Make sure the dataframe can handle multiindices
        if multiindex:
            df = homogenize_multiindex(df, "Input Data")

        # Create dataframe with the new features
        res = pd.DataFrame(features, index=df.index, columns=labels)

        if inplace:
            # Update the existing dataframe
            df[labels] = res[labels]
            return None
        else:
            # Create new dataframe and ensure columns are ordered properly
            res_labels = res.columns.tolist()
            if overwrite:
                overlapping_labels = [c for c in res_labels if c in df.columns]
                df = df.drop(columns=overlapping_labels)
            new = pd.concat([df, res], axis=1)
            return new[df.columns.tolist() + res.columns.tolist()]

    def _generate_column_labels(self, multiindex, return_errors):
        """Create a list of column names for a dataframe

        Args:
            multiindex (bool): Whether the dataframe has a multiindex
            return_errors (bool): Whether the dataframe will include columns
        Returns:
            list of column names for the dataframe
        """
        # Get the names of the features
        labels = self.feature_labels()

        # Add columns for the errors from the featurizer
        if return_errors:
            labels.append(self.__class__.__name__ + " Exceptions")

        ix_types = (pd.Index, list, tuple)
        if multiindex and len(labels[0]) == 2 and isinstance(labels[0], ix_types):
            # conversion featurizer, aiming to featurize in place.
            # conversion featurizers only have one feature label.
            # If return_errors=False, the transformation is:
            # [('l1', 'l2')] -> [('l1', 'l2')] (i.e. unaltered).
            # But if return_errors=True, the transformation is:
            # [('l1', 'l2'), 'feat Exceptions'] ->
            # [('l1', 'l2'), ('l1', 'feat Exceptions')]
            tmp_labels = [label if isinstance(label, str) else label[1]
                          for label in labels]
            indices = ([labels[0][0]], tmp_labels)
            labels = pd.MultiIndex.from_product(indices)

        elif multiindex:
            indices = ([self.__class__.__name__], labels)
            labels = pd.MultiIndex.from_product(indices)
        return labels

    def featurize_many(self, entries, ignore_errors=False, return_errors=False,
                       pbar=True):
        """Featurize a list of entries.

        If `featurize` takes multiple inputs, supply inputs as a list of tuples.

        Featurize_many supports entries as a list, tuple, numpy array,
        Pandas Series, or Pandas DataFrame.

        Args:
            entries (list-like object): A list of entries to be featurized.
            ignore_errors (bool): Returns NaN for entries where exceptions are
                thrown if True. If False, exceptions are thrown as normal.
            return_errors (bool): If True, returns the feature list as
                determined by ignore_errors with traceback strings added
                as an extra 'feature'. Entries which featurize without
                exceptions have this extra feature set to NaN.
            pbar (bool): Show a progress bar for featurization if True.

        Returns:
            (list) features for each entry.
        """

        if return_errors and not ignore_errors:
            raise ValueError("Please set ignore_errors to True to use"
                             " return_errors.")

        # Check inputs
        if not isinstance(entries, (tuple, list, np.ndarray, pd.Series, pd.DataFrame)):
            raise Exception("'entries' must be a list-like object")

        # Special case: Empty list
        if len(entries) == 0:
            return []

        # If the featurize function only has a single arg, zip the inputs
        if isinstance(entries, pd.DataFrame):
            entries = entries.values
        elif isinstance(entries, pd.Series) or not isinstance(entries[0], (tuple, list, np.ndarray)):
            entries = zip(entries)

        # Add a progress bar
        if pbar:
            # list() required, tqdm has issues with memory if generator given
            entries = tqdm(list(entries), desc=self.__class__.__name__)

        # Run the actual featurization
        if self.n_jobs == 1:
            return [self.featurize_wrapper(x, ignore_errors=ignore_errors,
                                           return_errors=return_errors) for x in entries]
        else:
            if sys.version_info[0] < 3:
                warnings.warn("Multiprocessing is not supported in "
                              "matminer for Python 2.x. Multiprocessing has "
                              "been disabled. Please upgrade to Python 3.x to "
                              "enable multiprocessing.")

                self.set_n_jobs(1)
                return self.featurize_many(entries,
                                           ignore_errors=ignore_errors,
                                           return_errors=return_errors,
                                           pbar=pbar)
            with Pool(self.n_jobs) as p:
                func = partial(self.featurize_wrapper,
                               return_errors=return_errors,
                               ignore_errors=ignore_errors)
                return p.map(func, entries, chunksize=self.chunksize)

    def featurize_wrapper(self, x, return_errors=False, ignore_errors=False):
        """
        An exception wrapper for featurize, used in featurize_many and
        featurize_dataframe. featurize_wrapper changes the behavior of featurize
        when ignore_errors is True in featurize_many/dataframe.

        Args:
             x: input data to featurize (type depends on featurizer).
             ignore_errors (bool): Returns NaN for entries where exceptions are
                thrown if True. If False, exceptions are thrown as normal.
             return_errors (bool): If True, returns the feature list as
                determined by ignore_errors with traceback strings added
                as an extra 'feature'. Entries which featurize without
                exceptions have this extra feature set to NaN.

        Returns:
            (list) one or more features.
        """
        try:
            # Successful featurization returns nan for an error.
            if return_errors:
                # Append operation must be agnostic to both ndarrays and lists
                return list(self.featurize(*x)) + [float("nan")]
            else:
                return self.featurize(*x)
        except BaseException as e:
            if ignore_errors:
                if return_errors:
                    features = [float("nan")] * len(self.feature_labels())
                    error = traceback.format_exception(*sys.exc_info())
                    return features + ["".join(error)]
                else:
                    return [float("nan")] * len(self.feature_labels())
            else:
                msg = str(e)
                msg += "\nTO SKIP THESE ERRORS when featurizing specific " \
                       "compounds, set 'ignore_errors=True' when running " \
                       "the batch featurize() operation (e.g., " \
                       "featurize_many(), featurize_dataframe(), etc.)."
                reraise(type(e), type(e)(msg), sys.exc_info()[2])

    @abstractmethod
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

    @abstractmethod
    def feature_labels(self):
        """
        Generate attribute names.

        Returns:
            ([str]) attribute labels.
        """

        raise NotImplementedError("feature_labels() is not defined!")

    @abstractmethod
    def citations(self):
        """
        Citation(s) and reference(s) for this feature.

        Returns:
            (list) each element should be a string citation,
                ideally in BibTeX format.
        """

        raise NotImplementedError("citations() is not defined!")

    @abstractmethod
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
    Class to run multiple featurizers on the same input data.

    All featurizers must take the same kind of data as input
    to the featurize function.

    Args:
        featurizers (list of BaseFeaturizer): A list of featurizers to run.
        iterate_over_entries (bool): Whether to iterate over the entries or
            featurizers. Iterating over entries will enable increased caching
            but will only display a single progress bar for all featurizers.
            If set to False, iteration will be performed over featurizers,
            resulting in reduced caching but individual progress bars for each
            featurizer.
    """

    def __init__(self, featurizers, iterate_over_entries=True):
        self.featurizers = featurizers
        self.iterate_over_entries = iterate_over_entries

    def featurize(self, *x):
        return [feature for f in self.featurizers
                for feature in f.featurize(*x)]

    def feature_labels(self):
        return sum([f.feature_labels() for f in self.featurizers], [])

    def fit(self, X, y=None, **fit_kwargs):
        for f in self.featurizers:
            f.fit(X, y, **fit_kwargs)
        return self

    def featurize_many(self, entries, ignore_errors=False, return_errors=False,
                       pbar=True):
        if self.iterate_over_entries:
            return super(MultipleFeaturizer, self).featurize_many(
                entries, ignore_errors=ignore_errors,
                return_errors=return_errors, pbar=pbar)
        else:
            features = [f.featurize_many(entries, ignore_errors=ignore_errors,
                                         return_errors=return_errors, pbar=pbar)
                        for f in self.featurizers]
            return [sum(x, []) for x in zip(*features)]

    def featurize_wrapper(self, x, return_errors=False, ignore_errors=False):
        if self.iterate_over_entries:
            return [feature for f in self.featurizers for feature in
                    f.featurize_wrapper(x, return_errors=return_errors,
                                        ignore_errors=ignore_errors)]
        else:
            return super(MultipleFeaturizer, self).featurize_wrapper(
                x, return_errors=return_errors, ignore_errors=ignore_errors)

    def citations(self):
        return list(set(sum([f.citations() for f in self.featurizers], [])))

    def implementors(self):
        return list(set(sum([f.implementors() for f in self.featurizers], [])))

    def _generate_column_labels(self, multiindex, return_errors):
        return np.hstack([f._generate_column_labels(multiindex, return_errors)
                          for f in self.featurizers])

    def set_n_jobs(self, n_jobs):
        super(MultipleFeaturizer, self).set_n_jobs(n_jobs)
        for featurizer in self.featurizers:
            featurizer.set_n_jobs(n_jobs)


class StackedFeaturizer(BaseFeaturizer):
    """
    Use the output of a machine learning model as features

    For regression models, we use the single output class.

    For classification models, we use the probability for the first N-1 classes where N is the
    number of classes.
    """

    def __init__(self, featurizer=None, model=None, name=None,
                 class_names=None):
        """Initialize featurizer

        Args:
            featurizer (BaseFeaturizer): Featurizer used to generate inputs to the model
            model (BaseEstimator): Fitted machine learning model to be evaluated
            name (str): [Optional] name of model, used when creating feature names
                class_names ([str]): Required for classification models, used when creating
                feature names (scikit-learn does not specify the number of classes for
                a classifier). Class names must be in the same order as the classes in the model
                (e.g., class_names[0] must be the name of the class 0)
        """

        # Store settings
        self.name = name
        self.class_names = class_names
        self.featurizer = featurizer
        self.model = model

        # Present warning about class_names
        if self.class_names is None and self._is_classifier():
            warnings.warn(
                'Class names are required for featurize_dataframe and feature_labels')

    def _is_classifier(self):
        """Whether the underlying model is a classifier

        Return:
            (boolean) whether `self.model` is a classifier
        """
        return is_classifier(self.model) or hasattr(self.model, 'predict_proba')

    def featurize(self, *x):
        # Generate the features
        # TODO: Explore checking whether features have already been computed. Feature for MultiFeaturizer? -lw
        features = [self.featurizer.featurize(*x)]

        # Run the model
        if self._is_classifier():
            output = self.model.predict_proba(features)[0]
            return output[:-1]
        else:
            return [self.model.predict(features)]

    def feature_labels(self):
        name = self.name or ''
        if self._is_classifier():
            if self.class_names is None:
                raise ValueError(
                    'Class names are required for classification models')
            return ['{} P({})'.format(name, cn).lstrip() for cn in
                    self.class_names[:-1]]
        else:
            return ['{} prediction'.format(name).lstrip()]

    def implementors(self):
        return ['Logan Ward']

    def citations(self):
        return []
