from __future__ import division, unicode_literals

import sys, traceback
import warnings
import pandas as pd
import numpy as np
from six import string_types, reraise
from multiprocessing import Pool, cpu_count

from sklearn.base import TransformerMixin, BaseEstimator, is_classifier
from matminer.utils.utils import homogenize_multiindex


class BaseFeaturizer(BaseEstimator, TransformerMixin):
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

    You can also employ the featurizer as part of a ScikitLearn Pipeline object.
    For these cases, scikit-learn calls the `transform` function of the
    `BaseFeaturizer` which is a less-featured wrapper of `featurize_many`. You
    would then provide your input data as an array to the Pipeline, which would
    output the featurers as an array.

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
        `implementors` - Returns a list of people who contributed writing a paper

    None of these operations should change the state of the featurizer. I.e.,
    running each method twice should no produce different results, no class
    attributes should be changed, unning one operation should not affect the
    output of another.

    All options of the featurizer must be set by the `__init__` function. All
    options must be listed as keyword arguments with default values, and the
    value must be saved as a class attribute with the same name (e.g., argument
    `n` should be stored in `self.n`). These requirements are necessary for
    compatibility with the `get_params` and `set_params` methods of
    `BaseEstimator`, which enable easy interoperability with scikit-learn.

    Depending on the complexity of your featurizer, it may be worthwhile to
    implement a `from_preset` class method. The `from_preset` method takes the
    name of a preset and returns an instance of the featurizer with some
    hard-coded set of inputs. The `from_preset` option is particularly useful
    for defining the settings used by papers in the literature.

    Optionally, you can implement the `fit` operation if there are attributes of
    your featurizer that must be set for the featurizer to work. Any variables
    that are set by fitting should be stored as class attributes that end with
    an underscore. (This follows the pattern used by scikit-learn).

    Another implementation to consider is whether it is worth making any utility
    operations for your featurizer. `featurize` must return a list of features,
    but this may not be the most natural representation for your features (e.g.,
    a `dict` could be better). Making a separate function for computing features
    in this natural representation and having the `featurize` function call this
    method and then convert the data into a list is a recommended approach.
    Users who want to compute the representation in the natural form can use the
    utility function and users who want the data in a ML-ready format (list) can
    call `featurize`. See `PartialRadialDistributionFunction` for an example of
    this concept.

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
        """Set the number of threads for this """
        self._n_jobs = n_jobs

    @property
    def n_jobs(self):
        return self._n_jobs if hasattr(self, '_n_jobs') else cpu_count()

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

        return self.featurize_many(X, ignore_errors=True)

    def fit_featurize_dataframe(self, df, col_id, *args, **kwargs):
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

        Returns:
            updated dataframe based on featurizer fitted to that dataframe.
        """
        return self.fit(df[col_id]).featurize_dataframe(df, col_id, *args,
                                                        **kwargs)

    def featurize_dataframe(self, df, col_id, ignore_errors=False,
                            return_errors=False, inplace=True,
                            multiindex=False):
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
            inplace (bool): Whether to add new columns to input dataframe (df)

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
        features = self.featurize_many(df[col_id].values,
                                       ignore_errors=ignore_errors,
                                       return_errors=return_errors)
        if return_errors:
            labels.append(self.__class__.__name__ + " Exceptions")

        if multiindex:
            indices = ([self.__class__.__name__], labels)
            labels = pd.MultiIndex.from_product(indices)
            df = homogenize_multiindex(df, "Input Data")
        elif isinstance(df.columns, pd.MultiIndex):
            # If input df is multi, but multi not enabled...
            raise ValueError("Please enable multiindexing to featurize an input"
                             " dataframe containing a column multiindex.")

        # Create dataframe with the new features
        res = pd.DataFrame(features, index=df.index, columns=labels)

        if inplace:
            # Update the existing dataframe
            for k in labels:
                df[k] = res[k]
            return df
        else:
            # Create new dataframe and ensure columns are ordered properly
            new = pd.concat([df, res], axis=1)
            return new[df.columns.tolist() + res.columns.tolist()]

    def featurize_many(self, entries, ignore_errors=False, return_errors=False):
        """
        Featurize a list of entries.
        If `featurize` takes multiple inputs, supply inputs as a list of tuples.

        Args:
            entries (list): A list of entries to be featurized.
            ignore_errors (bool): Returns NaN for entries where exceptions are
                thrown if True. If False, exceptions are thrown as normal.
            return_errors (bool): If True, returns the feature list as
                determined by ignore_errors with traceback strings added
                as an extra 'feature'. Entries which featurize without
                exceptions have this extra feature set to NaN.

        Returns:
            (list) features for each entry.
        """

        if return_errors and not ignore_errors:
            raise ValueError("Please set ignore_errors to True to use"
                             " return_errors.")

        self.__ignore_errors = ignore_errors
        self.__return_errors = return_errors

        # Check inputs
        if not hasattr(entries, '__getitem__'):
            raise Exception("'entries' must be a list-like object")

        # Special case: Empty list
        if len(entries) is 0:
            return []

        # If the featurize function only has a single arg, zip the inputs
        if not isinstance(entries[0], (tuple, list, np.ndarray)):
            entries = zip(entries)

        # Run the actual featurization
        if self.n_jobs == 1:
            return [self.featurize_wrapper(x) for x in entries]
        else:
            if sys.version_info[0] < 3:
                warnings.warn("Multiprocessing is not supported in "
                              "matminer for Python 2.x. Multiprocessing has "
                              "been disabled. Please upgrade to Python 3.x to "
                              "enable multiprocessing.")

                self.set_n_jobs(1)
                return self.featurize_many(entries,
                                           ignore_errors=ignore_errors,
                                           return_errors=return_errors)
            with Pool(self.n_jobs) as p:
                return p.map(self.featurize_wrapper, entries)

    def featurize_wrapper(self, x):
        """
        An exception wrapper for featurize, used in featurize_many and
        featurize_dataframe. featurize_wrapper changes the behavior of featurize
        when ignore_errors is True in featurize_many/dataframe.

        Args:
             x: input data to featurize (type depends on featurizer).

        Returns:
            (list) one or more features.
        """
        try:
            # Successful featurization returns nan for an error.
            if self.__return_errors:
                return self.featurize(*x) + [float("nan")]
            else:
                return self.featurize(*x)
        except BaseException as e:
            if self.__ignore_errors:
                if self.__return_errors:
                    features = [float("nan")] * len(self.feature_labels())
                    error = traceback.format_exception(*sys.exc_info())
                    return features + ["".join(error)]
                else:
                    return [float("nan")] * len(self.feature_labels())
            else:
                msg = str(e)
                msg += "\nTo skip errors when featurizing specific compounds," \
                       " consider running the batch featurize() operation " \
                       "(e.g., featurize_many(), featurize_dataframe(), etc.)" \
                       " with ignore_errors=True"
                reraise(type(e), type(e)(msg), sys.exc_info()[2])

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
        return np.hstack(np.squeeze(f.featurize(*x)) for f in self.featurizers)

    def set_n_jobs(self, n_jobs):
        for f in self.featurizers:
            f.set_n_jobs(n_jobs)

    def feature_labels(self):
        return sum([f.feature_labels() for f in self.featurizers], [])

    def fit_featurize_dataframe(self, df, col_id, *args, **kwargs):
        for f in self.featurizers:
            f.fit(df[col_id])
        return self.featurize_dataframe(df, col_id, *args, **kwargs)

    def featurize_dataframe(self, df, col_id, ignore_errors=False,
                            return_errors=False, inplace=True,
                            multiindex=False):
        """
        Featurize dataframe is overloaded in order to allow
        compatibility with Featurizers that overload featurize_dataframe
        """

        if multiindex:
            if not isinstance(df.columns, pd.MultiIndex):
                col_id = ("Input Data", col_id)
            df = homogenize_multiindex(df, "Input Data")

        # Detect if any featurizers override featurize_dataframe
        override = ["featurize_dataframe" in f.__class__.__dict__.keys()
                    for f in self.featurizers]
        if any(override):
            warnings.warn(
                "One or more featurizers overrides featurize_dataframe, "
                "featurization will be sequential and may diminish performance")

        for f in self.featurizers:
            df = f.featurize_dataframe(df, col_id, ignore_errors,
                                       return_errors, inplace, multiindex)

            if multiindex:
                feature_labels = [(f.__class__.__name__, flabel) for flabel in f.feature_labels()]
            else:
                feature_labels = f.feature_labels()
            df[feature_labels] = df[feature_labels].applymap(np.squeeze)
        return df

    def citations(self):
        return list(set(sum([f.citations() for f in self.featurizers], [])))

    def implementors(self):
        return list(set(sum([f.implementors() for f in self.featurizers], [])))


class StackedFeaturizer(BaseFeaturizer):
    """Use the output of a machine learning model as features

    For regression models, we use the single output class.

    For classification models, we use the probability for the first N-1 classes where N is the
    number of classes.
    """

    def __init__(self, featurizer=None, model=None, name=None, class_names=None):
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
            print('WARNING: Class names are required for featurize_dataframe and feature_labels',
                  file=sys.stderr)

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
                raise ValueError('Class names are required for classification models')
            return ['{} P({})'.format(name, cn).lstrip() for cn in self.class_names[:-1]]
        else:
            return ['{} prediction'.format(name).lstrip()]

    def implementors(self):
        return ['Logan Ward']

    def citations(self):
        return []
