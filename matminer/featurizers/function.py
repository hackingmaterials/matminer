from __future__ import division

import numpy as np
from sympy.parsing.sympy_parser import parse_expr
import sympy as sp
import itertools
from six import string_types
from pandas import DataFrame, Series

from collections import OrderedDict

from matminer.featurizers.base import BaseFeaturizer
from sklearn.exceptions import NotFittedError


# Default expressions to include in function featurizer
default_exps = ["x", "1/x", "sqrt(x)", "1/sqrt(x)", "x**2", "x**-2", "x**3",
                "x**-3", "log(x)", "1/log(x)", "exp(x)", "exp(-x)"]


class FunctionFeaturizer(BaseFeaturizer):
    """
    Features from functions applied to existing features, e.g. "1/x"

    This featurizer must be fit either by calling .fit_featurize_dataframe
    or by calling .fit followed by featurize_dataframe.

    This class featurizes a dataframe according to a set
    of expressions representing functions to apply to
    existing features. The approach here has uses a sympy-based
    parsing of string expressions, rather than explicit
    python functions.  The primary reason this has been
    done is to provide for better support for book-keeping
    (e. g. with feature labels), substitution, and elimination
    of symbolic redundancy, which sympy is well-suited for.

    Args:
        expressions ([str]): list of sympy-parseable expressions
            representing a function of a single variable x, e. g.
            ["1 / x", "x ** 2"], defaults to the list above
        multi_feature_depth (int): how many features to include if using
            multiple fields for functionalization, e. g. 2 will
            include pairwise combined features
        postprocess (function or type): type to cast functional outputs
            to, if, for example, you want to include the possibility of
            complex numbers in your outputs, use postprocess=np.complex,
            defaults to float
        combo_function (function): function to combine multi-features,
            defaults to np.prod (i.e. cumulative product of expressions),
            note that a combo function must cleanly process sympy
            expressions and **takes a list of arbitrary length as input**,
            other options include np.sum
        latexify_labels (bool): whether to render labels in latex,
            defaults to False
    """

    def __init__(self, expressions=None, multi_feature_depth=1,
                 postprocess=None, combo_function=None,
                 latexify_labels=False):

        self.expressions = expressions or default_exps
        self.multi_feature_depth = multi_feature_depth
        self.combo_function = combo_function or np.prod
        self.latexify_labels = latexify_labels
        self.postprocess = postprocess or float
        self._feature_labels = None

    @property
    def exp_dict(self):
        """
        Generates a dictionary of expressions keyed by number of
        variables in each expression

        Returns:
            Dictionary of expressions keyed by number of variables
        """
        # Generate lists of sympy expressions keyed by number of features
        return OrderedDict(
            [(n, generate_expressions_combinations(self.expressions, n,
                                                   self.combo_function))
             for n in range(1, self.multi_feature_depth+1)])

    def featurize(self, *args):
        """
        Main featurizer function, essentially iterates over all
        of the functions in self.function_list to generate
        features for each argument.

        Args:
            *args: list of numbers to generate functional output
                features

        Returns:
            list of functional outputs corresponding to input args
        """
        return list(self._exp_iter(*args, postprocess=self.postprocess))

    def feature_labels(self):
        """
        Returns:
            Set of feature labels corresponding to expressions
        """
        if not self._feature_labels:
            raise NotFittedError("Feature labels is only set if data is fitted"
                                 " to a dataframe")
        return self._feature_labels

    def fit(self, X, y=None, **fit_kwargs):
        """
        Sets the feature labels.  Not intended to be used by a user,
        only intended to be invoked as part of featurize_dataframe

        Args:
            X (DataFrame or array-like): data to fit to

        Returns:
            Set of feature labels corresponding to expressions
        """
        if isinstance(X, DataFrame):
            self._feature_labels = self.generate_string_expressions(X.columns)
        elif isinstance(X, Series):
            self._feature_labels = self.generate_string_expressions(X.name)
        return self

    def generate_string_expressions(self, input_variable_names):
        """
        Method to generate string expressions for input strings,
        mainly used to generate columns names for featurize_dataframe

        Args:
            input_variable_names ([str]): strings corresponding to
                functional input variable names

        Returns:
            List of string expressions generated by substitution of
            variable names into functions
        """
        if isinstance(input_variable_names, string_types):
            input_variable_names = [input_variable_names]
        postprocess = sp.latex if self.latexify_labels else str
        return list(self._exp_iter(*input_variable_names,
                                   postprocess=postprocess))

    def _exp_iter(self, *args, postprocess=None):
        """
        Generates an iterator for substitution of a set
        of args into the set of expression corresponding
        to the featurizer, intended primarily to remove
        replicated code in featurize and feature labels

        Args:
            *args: args to loop over combinations and substitions for
            postprocess (function): postprocessing function, e. g.
                to cast to another type, float, str

        Returns:
            iterator for all substituted expressions

        """
        postprocess = postprocess or (lambda x: x)
        for n in range(1, self.multi_feature_depth + 1):
            for arg_combo in itertools.combinations(args, n):
                subs_dict = {"x{}".format(m): arg
                             for m, arg in enumerate(arg_combo)}
                for exp in self.exp_dict[n]:
                    # TODO: this is a workaround for the problem
                    # TODO: postprocessing functional incompatility,
                    # TODO: e. g. sqrt(-1), 1 / 0
                    try:
                        yield postprocess(exp.subs(subs_dict))
                    except (TypeError, ValueError):
                        yield None

    def citations(self):
        return ["@article{Ramprasad2017,"
                "author = {Ramprasad, Rampi and Batra, Rohit and "
                           "Pilania, Ghanshyam and Mannodi-Kanakkithodi, Arun "
                           "and Kim, Chiho},"
                "doi = {10.1038/s41524-017-0056-5},"
                "journal = {npj Computational Materials},"
                "title = {Machine learning in materials informatics: recent "
                          "applications and prospects},"
                "volume = {3},number={1}, pages={54}, year={2017}}"]

    def implementors(self):
        return ['Joseph Montoya']


# TODO: Have this filter expressions that evaluate to things without vars,
# TODO:      # e. g. floats/ints
def generate_expressions_combinations(expressions, combo_depth=2,
                                      combo_function=np.prod):
    """
    This function takes a list of strings representing functions
    of x, converts them to sympy expressions, and combines
    them according to the combo_depth parameter.  Also filters
    resultant expressions for any redundant ones determined
    by sympy expression equivalence.

    Args:
        expressions (strings): all of the sympy-parseable strings
            to be converted to expressions and combined, e. g.
            ["1 / x", "x ** 2"], must be functions of x
        combo_depth (int): the number of independent variables to consider
        combo_function (method): the function which combines the
            the respective expressions provided, defaults to np.prod,
            i. e. the cumulative product of the expressions

    Returns:
        list of unique non-trivial expressions for featurization
            of inputs
    """
    # Convert to array for simpler subsitution
    exp_array = sp.Array([parse_expr(exp) for exp in expressions])

    # Generate all of the combinations
    combo_exps = []
    all_arrays = [exp_array.subs({"x": "x{}".format(n)})
                  for n in range(combo_depth)]
    # Get all sets of expressions
    for exp_set in itertools.product(*all_arrays):
        # Get all permutations of each set
        for exp_perm in itertools.permutations(exp_set):
            combo_exps.append(combo_function(exp_perm))

    # Filter for unique combinations, also remove identity
    unique_exps = list(set(combo_exps) - {parse_expr('x0')})
    # Sort to keep ordering
    unique_exps = sorted(unique_exps, key=lambda x: combo_exps.index(x))
    return unique_exps