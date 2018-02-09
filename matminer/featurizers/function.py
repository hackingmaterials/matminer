from __future__ import division

"""
This module includes code to featurize data based on
common mathematical expressions used as features.
"""

import numpy as np
from sympy.parsing.sympy_parser import parse_expr
import itertools

from matminer.featurizers.base import BaseFeaturizer

# This serves as a list of default functions
default_functions = [lambda x: x,
                     lambda x: 1 / x,
                     lambda x: x ** 0.5,
                     lambda x: x ** -0.5,
                     lambda x: x ** 2,
                     lambda x: x ** -2,
                     lambda x: x ** 3,
                     lambda x: x ** -3,
                     lambda x: log(x),
                     lambda x: 1 / log(x),
                     lambda x: exp(x),
                     lambda x: exp(-x)]

default_exps = ["x", "1/x", "sqrt(x)", "1/sqrt(x)", "x**2", "x**-2", "x**3",
                "x**-3", "log(x)", "1/log(x)", "exp(x)", "exp(-x)"]

def multiply_function_outputs(functions, inputs):
    """

    Args:
        functions:
        inputs:

    Returns:

    """
    return np.prod([f(i) for f, i in zip(functions, inputs)])


class FunctionFeaturizer(BaseFeaturizer):
    """
    This class featurizes a dataframe according to a set
    of functions for existing features
    """

    def __init__(self, function_list=None, multi_feature_depth=1,
                 combo_function=None):

        """
        Args:
            function_list: list of functions to use for featurization
            multi_feature_depth: how many features to include if using
                multiple fields for functionalization, e. g. 2 will
                include pairwise combined features
            combo_function: function to combine multi-features, defaults
                to product (i. e. multiplying each function), note
                that a combo function must take two arguments, a list
                of functions and a list of inputs, see
                multiply_function_outputs above for an example.
        """
        self.function_list = function_list or default_functions
        self.multi_feature_depth = multi_feature_depth
        self.combo_function = combo_function or multiply_function_outputs


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
        # Start with single features
        features = []
        for arg in args:
            for function in self.function_list:
                features.append(function(arg))

        for combo_population in range(2, self.multi_feature_depth + 1):
            for inputs in itertools.combinations(args, combo_population):
                for funcs in itertools.combinations(
                        self.function_list, combo_population):
                    features.append(self.combo_function(funcs, inputs))

        return features


    def feature_labels(self, col_id):
        if isinstance(col_id, str):
            col_id = [col_id]



    def citations(self):
        return ["@article{Ramprasad2017,"
                "author = {Ramprasad, Rampi and Batra, Rohit and Pilania, Ghanshyam"
                          "and Mannodi-Kanakkithodi, Arun and Kim, Chiho,"
                "doi = {10.1038/s41524-017-0056-5},"
                "journal = {npj Computational Materials},"
                "title = {Machine learning in materials informatics: recent applications and prospects},"
                "volume = {3},number={1}, pages={54}, year={2017}}"]

    def implementors(self):
        return ['Joseph Montoya']

