from __future__ import unicode_literals, division, print_function

import unittest
import pandas as pd

from matminer.featurizers.function import FunctionFeaturizer, \
    generate_expressions_combinations
import numpy as np
from sympy.parsing.sympy_parser import parse_expr


class TestFunctionFeaturizer(unittest.TestCase):

    def setUp(self):
        self.test_df = pd.DataFrame(
            [{"a": n, "b": n+1, "c": n+2} for n in range(-1, 10)])


    def test_featurize(self):
        ff = FunctionFeaturizer()
        # Test basic default functionality
        new_df = ff.featurize_dataframe(self.test_df, 'a', inplace=False)
        # ensure NaN for undefined values
        self.assertTrue(pd.isnull(new_df['1/a'][1]))
        self.assertTrue(pd.isnull(new_df['sqrt(a)'][0]))

        # ensure function works correctly
        new_df = new_df.dropna()
        self.assertTrue(np.allclose(np.log(new_df['a']),
                                    new_df['log(a)']))

        # Test custom expression functionality
        expressions = ["1 / x"]
        ff = FunctionFeaturizer(expressions=expressions)
        new_df = ff.featurize_dataframe(self.test_df, 'a', inplace=False)
        new_df = new_df.dropna()
        self.assertTrue("1/a" in new_df.columns)
        self.assertFalse("a**2" in new_df.columns)
        self.assertTrue(np.allclose(1 / new_df['a'],
                                    new_df['1/a']))

        # Test multi-functionality
        ff = FunctionFeaturizer(expressions=expressions, multi_feature_depth=2)
        new_df = ff.featurize_dataframe(self.test_df, ['a', 'b'], inplace=False)
        new_df = new_df.dropna()
        self.assertTrue(np.allclose(new_df['1/a'] * new_df['1/b'],
                                    new_df['1/(a*b)']))

        ff = FunctionFeaturizer(expressions=expressions, multi_feature_depth=3)
        new_df = ff.featurize_dataframe(self.test_df, ['a', 'b', 'c'],
                                        inplace=False)
        new_df = new_df.dropna()
        self.assertTrue(np.allclose(new_df['1/a']*new_df['1/b']*new_df['1/c'],
                                    new_df['1/(a*b*c)']))

        # Test complex functionality
        expressions = ["sqrt(x)"]
        ff = FunctionFeaturizer(expressions=expressions, postprocess=np.complex)
        new_df = ff.featurize_dataframe(self.test_df, 'a', inplace=False)
        self.assertEqual(new_df['sqrt(a)'][0], 1j)


    def test_featurize_labels(self):
        # Test latexification
        ff = FunctionFeaturizer(latexify_labels=True)
        new_df = ff.featurize_dataframe(self.test_df, 'a', inplace=False)
        self.assertTrue("\sqrt{a}" in new_df.columns)


    def test_helper_functions(self):
        test_combo_1 = generate_expressions_combinations(["1 / x", "x ** 2"], 1)
        self.assertEqual(len(test_combo_1), 2)
        self.assertTrue(parse_expr("x0 ** 2") in test_combo_1)

        test_combo_2 = generate_expressions_combinations(["1 / x", "x ** 2"])
        self.assertTrue(parse_expr("1 / (x0 * x1)") in test_combo_2)
        self.assertTrue(parse_expr("x0 ** 2 * x1 ** 2") in test_combo_2)
        self.assertEqual(len(test_combo_2), 4)

        test_combo_3 = generate_expressions_combinations(["1 / x", "x ** 2"],
                                                         combo_depth=3)
        self.assertTrue(parse_expr("x0 ** 2 / (x1 * x2)") in test_combo_3)
        self.assertEqual(len(test_combo_3), 8)


if __name__ == "__main__":
    unittest.main()
