import copy
import unittest
import warnings
from itertools import product

import numpy as np
import pandas as pd
from pymatgen.core.structure import Structure
from pymatgen.util.testing import PymatgenTest
from sklearn.dummy import DummyClassifier, DummyRegressor

from matminer.featurizers.base import (
    BaseFeaturizer,
    MultipleFeaturizer,
    StackedFeaturizer,
)
from matminer.featurizers.structure import SiteStatsFingerprint
from matminer.utils.caching import _get_all_nearest_neighbors


class SingleFeaturizer(BaseFeaturizer):
    def feature_labels(self):
        return ["y"]

    def featurize(self, x):
        return [x + 1]

    def citations(self):
        return ["A"]

    def implementors(self):
        return ["Us"]


class SingleFeaturizerWithPrecheck(SingleFeaturizer):
    def precheck(self, x):
        # Return True if even and False if odd
        return True if x % 2 == 0 else False


class SingleFeaturizerMultiArgs(SingleFeaturizer):
    def featurize(self, *x):
        return [x[0] + x[1]]


class SingleFeaturizerMultiArgsWithPrecheck(SingleFeaturizerMultiArgs):
    def precheck(self, *x):
        # If the sum is even, return True, else False
        return True if (x[1] + x[0]) % 2 == 0 else False


class MultipleFeatureFeaturizer(BaseFeaturizer):
    def feature_labels(self):
        return ["w", "z"]

    def featurize(self, x):
        return [x - 1, x + 2]

    def citations(self):
        return ["A"]

    def implementors(self):
        return ["Them"]


class MatrixFeaturizer(BaseFeaturizer):
    def feature_labels(self):
        return ["representation"]

    def featurize(self, *x):
        return [np.eye(2, 2)]

    def citations(self):
        return ["C"]

    def implementors(self):
        return ["Everyone"]


class MultiArgs2(BaseFeaturizer):
    def __init__(self):
        self._featurizer = SingleFeaturizerMultiArgs()

    def featurize(self, *x):
        return self._featurizer.featurize(*x)

    def feature_labels(self):
        return ["y2"]

    def citations(self):
        return []

    def implementors(self):
        return []


class FittableFeaturizer(BaseFeaturizer):
    """
    This test featurizer tests fitting qualities of BaseFeaturizer, including
    refittability and different results based on different fits.
    """

    def fit(self, X, y=None, **fit_kwargs):
        self._features = ["a", "b", "c"][: len(X)]
        return self

    def featurize(self, x):
        return [x + 3, x + 4, 2 * x][: len(self._features)]

    def feature_labels(self):
        return self._features

    def citations(self):
        return ["Q"]

    def implementors(self):
        return ["A competing research group"]


class MultiTypeFeaturizer(BaseFeaturizer):
    """A featurizer that returns multiple dtypes"""

    def featurize(self, *x):
        return ["a", 1]

    def feature_labels(self):
        return ["label", "int_label"]

    def citations(self):
        return []

    def implementors(self):
        return []


class TestBaseClass(PymatgenTest):
    def setUp(self):
        self.single = SingleFeaturizer()
        self.multi = MultipleFeatureFeaturizer()
        self.matrix = MatrixFeaturizer()
        self.multiargs = SingleFeaturizerMultiArgs()
        self.fittable = FittableFeaturizer()

    @staticmethod
    def make_test_data():
        return pd.DataFrame({"x": [1, 2, 3]})

    def test_dataframe(self):
        data = self.make_test_data()
        data = self.single.featurize_dataframe(data, "x")
        self.assertArrayAlmostEqual(data["y"], [2, 3, 4])

        data = self.multi.featurize_dataframe(data, "x")
        self.assertArrayAlmostEqual(data["w"], [0, 1, 2])
        self.assertArrayAlmostEqual(data["z"], [3, 4, 5])

    def test_matrix(self):
        """Test the ability to add features that are matrices to a dataframe"""
        data = self.make_test_data()
        data = self.matrix.featurize_dataframe(data, "x")
        self.assertArrayAlmostEqual(np.eye(2, 2), data["representation"][0])

    def test_inplace(self):
        data = self.make_test_data()
        df_returned = self.single.featurize_dataframe(data, "x", inplace=False)
        self.assertNotIn("y", data.columns)
        self.assertIn("y", df_returned.columns)

        df_returned = self.single.featurize_dataframe(data, "x", inplace=True)
        self.assertIn("y", data)
        self.assertIsNone(df_returned)

    def test_indices(self):
        data = self.make_test_data()
        data.index = [4, 6, 6]

        data = self.single.featurize_dataframe(data, "x")
        self.assertArrayAlmostEqual(data["y"], [2, 3, 4])

    def test_multiple(self):
        # test iterating over both entries and featurizers
        for iter_entries in [True, False]:
            multi_f = MultipleFeaturizer([self.single, self.multi], iterate_over_entries=iter_entries)
            data = self.make_test_data()

            self.assertArrayAlmostEqual([2, 0, 3], multi_f.featurize(1))

            self.assertArrayEqual(["A"], multi_f.citations())

            implementors = multi_f.implementors()
            self.assertIn("Us", implementors)
            self.assertIn("Them", implementors)
            self.assertEqual(2, len(implementors))

            # Ensure BaseFeaturizer operation without overridden featurize_dataframe
            with warnings.catch_warnings(record=True) as w:
                data = multi_f.featurize_dataframe(data, "x")
                self.assertEqual(len(w), 0)
            self.assertArrayAlmostEqual(data["y"], [2, 3, 4])
            self.assertArrayAlmostEqual(data["w"], [0, 1, 2])
            self.assertArrayAlmostEqual(data["z"], [3, 4, 5])

            f = MatrixFeaturizer()
            multi_f = MultipleFeaturizer([self.single, self.multi, f])
            data = self.make_test_data()
            with warnings.catch_warnings(record=True) as w:
                data = multi_f.featurize_dataframe(data, "x")
                self.assertEqual(len(w), 0)

            self.assertArrayAlmostEqual(data["representation"][0], [[1.0, 0.0], [0.0, 1.0]])

    def test_multifeatures_multiargs(self):
        multiargs2 = MultiArgs2()

        # test iterating over both entries and featurizers
        for iter_entries in [True, False]:
            # Make a test dataset with two input variables
            data = self.make_test_data()
            data["x2"] = [4, 5, 6]

            # Create featurizer
            multi_f = MultipleFeaturizer([self.multiargs, multiargs2], iterate_over_entries=iter_entries)

            # Test featurize with multiple arguments
            features = multi_f.featurize(0, 2)
            self.assertArrayAlmostEqual([2, 2], features)

            # Test dataframe
            data = multi_f.featurize_dataframe(data, ["x", "x2"])
            self.assertEqual(["y", "y2"], multi_f.feature_labels())
            self.assertArrayAlmostEqual([[5, 5], [7, 7], [9, 9]], data[["y", "y2"]])
            # Test with multiindex
            data = multi_f.featurize_dataframe(data, ["x", "x2"], multiindex=True)
            self.assertIn(("MultiArgs2", "y2"), data.columns)
            self.assertIn(("SingleFeaturizerMultiArgs", "y"), data.columns)
            self.assertArrayAlmostEqual(
                [[5, 5], [7, 7], [9, 9]],
                data[[("SingleFeaturizerMultiArgs", "y"), ("MultiArgs2", "y2")]],
            )

    def test_featurize_many(self):
        # Single argument
        s = self.single
        s.set_n_jobs(2)
        mat = s.featurize_many([1, 2, 3])
        self.assertArrayAlmostEqual(mat, [[2], [3], [4]])

        # Multi-argument
        s = self.multiargs
        s.set_n_jobs(2)
        mat = s.featurize_many([[1, 4], [2, 5], [3, 6]])
        self.assertArrayAlmostEqual(mat, [[5], [7], [9]])

    def test_multiprocessing_df(self):
        # Single argument
        s = self.single
        data = self.make_test_data()
        s.set_n_jobs(2)
        data = s.featurize_dataframe(data, "x")
        self.assertArrayAlmostEqual(data["y"], [2, 3, 4])

        # Multi-argument
        s = self.multiargs
        data = self.make_test_data()
        s.set_n_jobs(2)
        data["x2"] = [4, 5, 6]
        data = s.featurize_dataframe(data, ["x", "x2"])
        self.assertArrayAlmostEqual(data["y"], [5, 7, 9])

    def test_fittable(self):
        data = self.make_test_data()
        ft = self.fittable

        # Test fit and featurize separately
        ft.fit(data["x"][:2])
        data = ft.featurize_dataframe(data, "x")
        self.assertArrayAlmostEqual(data["a"], [4, 5, 6])
        self.assertRaises(Exception, data.__getattr__, "c")

        # Test fit + featurize methods on new fits
        data = self.make_test_data()
        transformed = ft.fit_transform([data["x"][1]])
        self.assertArrayAlmostEqual(transformed[0], [5])
        data = self.make_test_data()
        data = ft.fit_featurize_dataframe(data, "x")
        self.assertArrayAlmostEqual(data["a"], [4, 5, 6])
        self.assertArrayAlmostEqual(data["b"], [5, 6, 7])
        self.assertArrayAlmostEqual(data["c"], [2, 4, 6])

    def test_stacked_featurizer(self):
        data = self.make_test_data()
        data["y"] = [1, 2, 3]

        # Test for a regressor
        model = DummyRegressor()
        model.fit(self.multi.featurize_many(data["x"]), data["y"])

        #  Test the predictions
        f = StackedFeaturizer(self.single, model)
        self.assertEqual([2], f.featurize(data["x"][0]))

        #  Test the feature names
        self.assertEqual(["prediction"], f.feature_labels())
        f.name = "ML"
        self.assertEqual(["ML prediction"], f.feature_labels())

        # Test classifier
        model = DummyClassifier(strategy="prior")
        data["y"] = [0, 0, 1]
        model.fit(self.multi.featurize_many(data["x"]), data["y"])

        #  Test the prediction
        f.model = model
        self.assertEqual([2.0 / 3], f.featurize(data["x"][0]))

        #  Test the feature labels
        self.assertRaises(ValueError, f.feature_labels)
        f.class_names = ["A", "B"]
        self.assertEqual(["ML P(A)"], f.feature_labels())

        # Test with three classes
        data["y"] = [0, 2, 1]
        model.fit(self.multi.featurize_many(data["x"]), data["y"])

        self.assertArrayAlmostEqual([1.0 / 3] * 2, f.featurize(data["x"][0]))
        f.class_names = ["A", "B", "C"]
        self.assertEqual(["ML P(A)", "ML P(B)"], f.feature_labels())

    def test_multiindex_inplace(self):
        df_1lvl = pd.DataFrame({"x": [1, 2, 3]})
        df_2lvl = pd.DataFrame({"x": [1, 2, 3]})
        df_2lvl.columns = pd.MultiIndex.from_product((["Custom"], df_2lvl.columns.values))
        df_3lvl = pd.DataFrame({"x": [1, 2, 3]})
        df_3lvl.columns = pd.MultiIndex.from_product((["Custom"], ["Custom2"], df_3lvl.columns.values))

        # If input dataframe has flat column index
        self.multi.featurize_dataframe(df_1lvl, "x", multiindex=True, inplace=True)
        self.assertEqual(df_1lvl[("Input Data", "x")].iloc[0], 1)
        self.assertEqual(df_1lvl[("MultipleFeatureFeaturizer", "w")].iloc[0], 0)

        # If input dataframe has 2-lvl column index
        self.multi.featurize_dataframe(df_2lvl, ("Custom", "x"), multiindex=True, inplace=True)
        self.assertEqual(df_2lvl[("Custom", "x")].iloc[0], 1)
        self.assertEqual(df_2lvl[("MultipleFeatureFeaturizer", "w")].iloc[0], 0)

        # If input dataframe has 2+ lvl column index
        with self.assertRaises(IndexError):
            self.multi.featurize_dataframe(df_3lvl, ("Custom", "Custom2", "x"), multiindex=True, inplace=True)

        # Make sure error is thrown when input df  is multiindexed, but multiindex not enabled
        df_compoundkey = pd.DataFrame({"x": [1, 2, 3]})
        df_compoundkey.columns = pd.MultiIndex.from_product((["CK"], df_compoundkey.columns.values))
        with self.assertRaises(ValueError):
            _ = self.multi.featurize_dataframe(df_compoundkey, ("CK", "x"))

    def test_multiindex_return(self):
        # For inplace=False, where the method of assigning keys is different
        df_1lvl = pd.DataFrame({"x": [1, 2, 3]})
        df_2lvl = pd.DataFrame({"x": [1, 2, 3]})
        df_2lvl.columns = pd.MultiIndex.from_product((["Custom"], df_2lvl.columns.values))
        df_3lvl = pd.DataFrame({"x": [1, 2, 3]})
        df_3lvl.columns = pd.MultiIndex.from_product((["Custom"], ["Custom2"], df_3lvl.columns.values))
        # If input dataframe has flat column index
        df_1lvl = self.multi.featurize_dataframe(df_1lvl, "x", inplace=False, multiindex=True)
        self.assertEqual(df_1lvl[("Input Data", "x")].iloc[0], 1)
        self.assertEqual(df_1lvl[("MultipleFeatureFeaturizer", "w")].iloc[0], 0)

        # If input dataframe has 2-lvl column index
        df_2lvl = self.multi.featurize_dataframe(df_2lvl, ("Custom", "x"), inplace=False, multiindex=True)
        self.assertEqual(df_2lvl[("Custom", "x")].iloc[0], 1)
        self.assertEqual(df_2lvl[("MultipleFeatureFeaturizer", "w")].iloc[0], 0)

        # If input dataframe has 2+ lvl column index
        with self.assertRaises(IndexError):
            _ = self.multi.featurize_dataframe(df_3lvl, ("Custom", "Custom2", "x"), inplace=False, multiindex=True)

    def test_multiindex_in_multifeaturizer(self):
        # Make sure multiplefeaturizer returns the correct sub-featurizer multiindex keys
        # test both iteration over entries and featurizers

        for iter_entries in [True, False]:
            mf = MultipleFeaturizer([self.multi, self.single], iterate_over_entries=iter_entries)

            df_1lvl = pd.DataFrame({"x": [1, 2, 3]})
            df_2lvl = pd.DataFrame({"x": [1, 2, 3]})
            df_2lvl.columns = pd.MultiIndex.from_product((["Custom"], df_2lvl.columns.values))
            df_3lvl = pd.DataFrame({"x": [1, 2, 3]})
            df_3lvl.columns = pd.MultiIndex.from_product((["Custom"], ["Custom2"], df_3lvl.columns.values))

            # If input dataframe has flat column index
            df_1lvl = mf.featurize_dataframe(df_1lvl, "x", multiindex=True)
            self.assertEqual(df_1lvl[("Input Data", "x")].iloc[0], 1)
            self.assertEqual(df_1lvl[("MultipleFeatureFeaturizer", "w")].iloc[0], 0)
            self.assertEqual(df_1lvl[("SingleFeaturizer", "y")].iloc[0], 2)

            # If input dataframe has 2-lvl column index
            df_2lvl = mf.featurize_dataframe(df_2lvl, ("Custom", "x"), multiindex=True)
            self.assertEqual(df_2lvl[("Custom", "x")].iloc[0], 1)
            self.assertEqual(df_2lvl[("MultipleFeatureFeaturizer", "w")].iloc[0], 0)
            self.assertEqual(df_2lvl[("SingleFeaturizer", "y")].iloc[0], 2)

            # If input dataframe has 2+ lvl column index
            with self.assertRaises(IndexError):
                df_3lvl = self.multi.featurize_dataframe(df_3lvl, ("Custom", "Custom2", "x"), multiindex=True)

    def test_caching(self):
        """Test whether MultiFeaturizer properly caches"""

        # have to iterate over entries to enable caching
        feat = MultipleFeaturizer(
            [
                SiteStatsFingerprint.from_preset("LocalPropertyDifference_ward-prb-2017"),
                SiteStatsFingerprint.from_preset("CoordinationNumber_ward-prb-2017"),
            ],
            iterate_over_entries=True,
        )

        # Reset the cache before tests
        _get_all_nearest_neighbors.cache_clear()

        # Create a dataframe with two SC structures in it
        data = pd.DataFrame(
            {
                "strcs": [
                    Structure([[3.52, 0, 0], [0, 3.52, 0], [0, 0, 3.52]], ["Al"], [[0, 0, 0]]),
                    Structure([[3.52, 0, 0], [0, 3.52, 0], [0, 0, 3.52]], ["Ni"], [[0, 0, 0]]),
                ]
            }
        )

        # Call featurize on both, check the number of cache misses/hits
        feat.featurize(data["strcs"][0])
        feat.featurize(data["strcs"][1])
        self.assertEqual(2, _get_all_nearest_neighbors.cache_info().hits)
        self.assertEqual(2, _get_all_nearest_neighbors.cache_info().misses)

        # Verify the number of cache misses, it should be the same as before
        feat.set_n_jobs(1)
        _get_all_nearest_neighbors.cache_clear()
        feat.featurize_dataframe(data, "strcs")

        self.assertEqual(2, _get_all_nearest_neighbors.cache_info().hits)
        self.assertEqual(2, _get_all_nearest_neighbors.cache_info().misses)

    def test_ignore_errors(self):
        # Make sure multiplefeaturizer returns the correct sub-featurizer multiindex keys

        # Iterate through many tests: single/parallel, returning errors or not,
        # multiindex or not, and interaction over entries/featurizers

        for mi, re, n, iter_entries in product([True, False], [True, False], [1, 2], [True, False]):

            mf = MultipleFeaturizer([self.multi, self.single], iterate_over_entries=iter_entries)
            # Make some test data that will cause errors
            data = pd.DataFrame({"x": ["a", 2, 3]})

            # Set the number of threads
            mf.set_n_jobs(n)

            # Make sure it completes successfully
            results = mf.featurize_many(data["x"], ignore_errors=True, return_errors=re)
            self.assertEqual(5 if re else 3, len(results[0]))

            # Make sure it works with featurize dataframe
            results = mf.featurize_dataframe(data, "x", ignore_errors=True, return_errors=re, multiindex=mi)
            self.assertEqual(6 if re else 4, len(results.columns))

            #  Special test for returning errors (only should work when returning errors)
            #   I only am going to test the single index case for simplicity
            if re and not mi:
                self.assertIn("TypeError", results.iloc[0]["SingleFeaturizer Exceptions"])

            # Make sure it throws an error
            with self.assertRaises(TypeError):
                mf.featurize_many([["a"], [1], [2]])

    def test_multitype_multifeat(self):
        """Test Multifeaturizer when a featurizer returns a non-numeric type"""

        # test both iteration over entries and featurizers
        for iter_entries in [True, False]:
            # Make the featurizer
            f = MultipleFeaturizer(
                [SingleFeaturizer(), MultiTypeFeaturizer()],
                iterate_over_entries=iter_entries,
            )
            f.set_n_jobs(1)

            # Make the test data
            data = self.make_test_data()

            # Add the columns
            data = f.featurize_dataframe(data, "x")

            # Make sure the types are as expected
            labels = f.feature_labels()
            self.assertArrayEqual(["int64", "object", "int64"], data[labels].dtypes.astype(str).tolist())
            self.assertArrayAlmostEqual(data["y"], [2, 3, 4])

    def test_multifeature_no_zero_index(self):
        """Test whether multifeaturizer can handle series that lack a entry
        with index==0
        """

        # Make a dataset without a index == 0
        data = pd.DataFrame({"x": [1], "y": [2]})
        data.index = [1]

        # Multifeaturize
        self.multiargs.set_n_jobs(1)
        self.single.featurize_many(data["x"])
        self.multiargs.featurize_many(data[["x", "y"]])

    def test_precheck(self):
        # Test with a single argument
        f = SingleFeaturizerWithPrecheck()
        data = self.make_test_data()
        self.assertTrue(f.precheck(2))
        self.assertFalse(f.precheck(3))
        frac = f.precheck_dataframe(data, "x", return_frac=True)
        self.assertAlmostEqual(frac, 1 / 3, places=4)

        data_ip = copy.deepcopy(data)
        f.precheck_dataframe(data_ip, "x", return_frac=False, inplace=True)
        pckey = "SingleFeaturizerWithPrecheck precheck pass"
        self.assertArrayEqual(data_ip[pckey].tolist(), [False, True, False])

        df2 = f.precheck_dataframe(data, "x", return_frac=False, inplace=False)
        self.assertArrayEqual(df2[pckey].tolist(), [False, True, False])
        self.assertNotIn(pckey, data.columns.tolist())

        # Test with multiple arguments
        fm = SingleFeaturizerMultiArgsWithPrecheck()
        self.assertTrue(fm.precheck(1, 1))
        self.assertFalse(fm.precheck(1, 2))

        data["x2"] = [3, 3, 3]
        frac = fm.precheck_dataframe(data, ["x", "x2"], return_frac=True)
        self.assertAlmostEqual(frac, 2 / 3)

        mpckey = "SingleFeaturizerMultiArgsWithPrecheck precheck pass"

        data_ipm = copy.deepcopy(data)
        fm.precheck_dataframe(data_ipm, ["x", "x2"], return_frac=False, inplace=True)
        self.assertArrayEqual(data_ipm[mpckey].tolist(), [True, False, True])

        df3 = fm.precheck_dataframe(data, ["x", "x2"], return_frac=False, inplace=False)
        self.assertArrayEqual(df3[mpckey].tolist(), [True, False, True])
        self.assertNotIn(mpckey, data.columns.tolist())


if __name__ == "__main__":
    unittest.main()
