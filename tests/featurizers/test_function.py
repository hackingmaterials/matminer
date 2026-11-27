import unittest

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sympy.parsing.sympy_parser import parse_expr

from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.function import (
    FunctionFeaturizer,
    generate_expressions_combinations,
)


class TestFunctionFeaturizer(unittest.TestCase):
    def setUp(self):
        self.test_df = pd.DataFrame([{"a": n, "b": n + 1, "c": n + 2} for n in range(-1, 10)])

    def test_featurize(self):
        ff = FunctionFeaturizer()
        # Test basic default functionality
        new_df = ff.fit_featurize_dataframe(self.test_df, "a", inplace=False)
        # ensure NaN for undefined values
        self.assertTrue(pd.isnull(new_df["1/a"][1]))
        self.assertTrue(pd.isnull(new_df["sqrt(a)"][0]))

        # ensure function works correctly
        new_df = new_df.dropna()
        self.assertTrue(np.allclose(np.log(new_df["a"]), new_df["log(a)"]))

        # Test custom expression functionality
        expressions = ["1 / x"]
        ff = FunctionFeaturizer(expressions=expressions)
        new_df = ff.fit_featurize_dataframe(self.test_df, "a", inplace=False)
        new_df = new_df.dropna()
        self.assertTrue("1/a" in new_df.columns)
        self.assertFalse("a**2" in new_df.columns)
        self.assertTrue(np.allclose(1 / new_df["a"], new_df["1/a"]))

        # Test multi-functionality
        ff = FunctionFeaturizer(expressions=expressions, multi_feature_depth=2)
        new_df = ff.fit_featurize_dataframe(self.test_df, ["a", "b"], inplace=False)
        new_df = new_df.dropna()
        self.assertTrue(np.allclose(new_df["1/a"] * new_df["1/b"], new_df["1/(a*b)"]))

        ff = FunctionFeaturizer(expressions=expressions, multi_feature_depth=3)
        new_df = ff.fit_featurize_dataframe(self.test_df, ["a", "b", "c"], inplace=False)
        new_df = new_df.dropna()
        self.assertTrue(np.allclose(new_df["1/a"] * new_df["1/b"] * new_df["1/c"], new_df["1/(a*b*c)"]))

        # Test complex functionality
        expressions = ["sqrt(x)"]
        ff = FunctionFeaturizer(expressions=expressions, postprocess=np.complex128)
        new_df = ff.fit_featurize_dataframe(self.test_df, "a", inplace=False)
        self.assertEqual(new_df["sqrt(a)"][0], 1j)

        ff = FunctionFeaturizer(expressions=expressions, multi_feature_depth=2, combo_function=np.sum)
        new_df = ff.fit_featurize_dataframe(self.test_df, ["a", "b"], inplace=False)
        self.assertAlmostEqual(new_df["sqrt(a) + sqrt(b)"][2], 2.41421356)

        # Test parallel vs. serial
        ff = FunctionFeaturizer()
        df = pd.DataFrame({"t2": [1, 2, 3]})

        ff.set_n_jobs(1)
        serial = ff.fit_featurize_dataframe(df, ["t2"], inplace=False)
        ff.set_n_jobs(2)
        parallel = ff.fit_featurize_dataframe(df, ["t2"], inplace=False)
        self.assertTrue(np.allclose(serial, parallel, equal_nan=True))

    def test_featurize_labels(self):
        # Test latexification
        ff = FunctionFeaturizer(latexify_labels=True)
        new_df = ff.fit_featurize_dataframe(self.test_df, "a", inplace=False)
        self.assertTrue(r"\sqrt{a}" in new_df.columns)
        # Ensure error is thrown with no labels
        ff = FunctionFeaturizer(latexify_labels=False)
        self.assertRaises(NotFittedError, ff.feature_labels)

    def test_featurize_literal_column_labels(self):
        values = [10.0, 20.0]
        ff = FunctionFeaturizer(expressions=["log(x)"])
        ff.set_n_jobs(1)

        for label, feature_label in (
            ("E", "log(E)"),
            ("N", "log(N)"),
            ("energy-gap", "log(energy-gap)"),
            ("energy gap", "log(energy_gap)"),
            (0, "log(0)"),
        ):
            with self.subTest(label=label):
                df = pd.DataFrame({label: values})
                result = ff.fit_featurize_dataframe(df, [label], pbar=False)
                self.assertEqual(list(result.columns), [label, feature_label])
                np.testing.assert_allclose(result[feature_label], np.log(values))

        self.assertEqual(ff.fit(pd.Series(values, name=0)).feature_labels(), ["log(0)"])

    def test_featurize_labels_matching_placeholders(self):
        ff = FunctionFeaturizer(expressions=["x"], multi_feature_depth=2)
        ff.set_n_jobs(1)
        df = pd.DataFrame({"x1": [2.0, 3.0], "b": [5.0, 7.0]})

        result = ff.fit_featurize_dataframe(df, ["x1", "b"], pbar=False)

        self.assertEqual(list(result.columns), ["x1", "b", "b*x1"])
        np.testing.assert_allclose(result["b*x1"], df["x1"] * df["b"])

    def test_helper_functions(self):
        test_combo_1 = generate_expressions_combinations(["1 / x", "x ** 2"], 1)
        self.assertEqual(len(test_combo_1), 2)
        self.assertTrue(parse_expr("x0 ** 2") in test_combo_1)

        test_combo_2 = generate_expressions_combinations(["1 / x", "x ** 2"])
        self.assertTrue(parse_expr("1 / (x0 * x1)") in test_combo_2)
        self.assertTrue(parse_expr("x0 ** 2 * x1 ** 2") in test_combo_2)
        self.assertEqual(len(test_combo_2), 4)

        test_combo_3 = generate_expressions_combinations(["1 / x", "x ** 2"], combo_depth=3)
        self.assertTrue(parse_expr("x0 ** 2 / (x1 * x2)") in test_combo_3)
        self.assertEqual(len(test_combo_3), 8)

        test_combo_4 = generate_expressions_combinations(["1 / x", "x ** 2", "exp(x)"], combo_function=np.sum)
        self.assertEqual(len(test_combo_4), 9)
        test_combo_4 = generate_expressions_combinations(
            ["1 / x", "x ** 2", "exp(x)"], combo_function=lambda x: x[1] - x[0]
        )
        self.assertEqual(len(test_combo_4), 18)

    def test_multi_featurizer(self):
        ff1 = FunctionFeaturizer(expressions=["x ** 2"])
        ff2 = FunctionFeaturizer(expressions=["exp(x)", "1 / x"])
        mf = MultipleFeaturizer([ff1, ff2])
        new_df = mf.fit_featurize_dataframe(self.test_df, ["a", "b", "c"], inplace=False)
        self.assertEqual(len(new_df), 11)


if __name__ == "__main__":
    unittest.main()
