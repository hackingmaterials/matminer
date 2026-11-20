from math import sqrt
from unittest import TestCase

import numpy as np

from matminer.featurizers.utils.stats import PropertyStats


class TestPropertyStats(TestCase):
    def setUp(self):
        self.sample_1 = [1] * 3
        self.sample_1_weights = [1] * 3
        self.sample_2 = [0.5, 1.5, 0]
        self.sample_2_weights = [2, 1, 0.5]

    def _run_test(self, statistic, sample_1, sample_1_weighted, sample_2, sample_2_weighted):
        """Run a test for a certain statistic against the two sample datasets

        Args:
            statistic: name of statistic
            sample_1: float, expected value for statistic of sample 1 without weights
            sample_1_weighted: float, expected value for statistic of sample 1 with weights
            sample_2: float, expected value for statistic of sample 2 without weights
            sample_2_weighted: float, expected value for statistic of sample 2 with weights
        """

        np.testing.assert_almost_equal(sample_1, PropertyStats.calc_stat(self.sample_1, statistic))
        np.testing.assert_almost_equal(
            sample_1_weighted,
            PropertyStats.calc_stat(self.sample_1, statistic, self.sample_1_weights),
        )
        np.testing.assert_almost_equal(sample_2, PropertyStats.calc_stat(self.sample_2, statistic))
        np.testing.assert_almost_equal(
            sample_2_weighted,
            PropertyStats.calc_stat(self.sample_2, statistic, self.sample_2_weights),
        )

    def test_minimum(self):
        self._run_test("minimum", 1, 1, 0, 0)

    def test_maximum(self):
        self._run_test("maximum", 1, 1, 1.5, 1.5)

    def test_range(self):
        self._run_test("range", 0, 0, 1.5, 1.5)

    def test_mean(self):
        self._run_test("mean", 1, 1, 2.0 / 3, 5.0 / 7)

    def test_avg_dev(self):
        self._run_test("avg_dev", 0, 0, 5.0 / 9, 0.448979592)

    def test_std_dev(self):
        self._run_test("std_dev", 0, 0, 0.623609564, 0.694365075)

    def test_skewness(self):
        self._run_test("skewness", np.nan, 0, 0.38180177, 0.559451361)

    def test_kurtosis(self):
        self._run_test("kurtosis", np.nan, 0, 1.5, 1.9403292181)

    def test_mode(self):
        self._run_test("mode", 1, 1, 0, 0.5)

        # Additional tests
        self.assertAlmostEqual(0, PropertyStats.mode([0, 1, 2], [1, 1, 1]))

    def test_holder_mean(self):
        self._run_test("holder_mean::0", 1, 1, np.prod(self.sample_2), 0)

        self._run_test("holder_mean::1", 1, 1, 2.0 / 3, 5.0 / 7)
        self._run_test("holder_mean::2", 1, 1, sqrt(5.0 / 6), 0.88640526)

        # Check harmonic means with nonzero inputs.
        self.assertAlmostEqual(PropertyStats.holder_mean([1, 1, 2], power=-1), 1.2, places=3)

        self.assertAlmostEqual(PropertyStats.holder_mean([1, 2], [2, 1], power=-1), 1.2, places=3)

    def test_weighted_harmonic_mean_with_zeros(self):
        cases = [
            ([0, 1], [1, 1], 0),
            ([0, 1], [0, 1], 1),
            ([0, 1, 2], [0, 2, 1], 1.2),
            ([1, 0], [1, 1], 0),
        ]
        for values, weights, expected in cases:
            for container in (list, np.array):
                with self.subTest(values=values, weights=weights, container=container):
                    self.assertAlmostEqual(
                        PropertyStats.calc_stat(container(values), "holder_mean::-1", container(weights)), expected
                    )

    def test_weighted_harmonic_mean_nonfinite(self):
        cases = [
            ([0, np.nan], [1, 1]),
            ([0, 1], [np.nan, 1]),
            ([0, 1], [1, np.inf]),
        ]
        for values, weights in cases:
            for reverse in (False, True):
                with self.subTest(values=values, weights=weights, reverse=reverse):
                    if reverse:
                        values, weights = values[::-1], weights[::-1]
                    with np.errstate(invalid="ignore"):
                        result = PropertyStats.holder_mean(np.array(values), np.array(weights), power=-1)
                    self.assertTrue(np.isnan(result))

    def test_geom_std_dev(self):
        # This is right. Yes, a list without variation has a geom_std_dev of 1
        self.assertAlmostEqual(1, PropertyStats.geom_std_dev([1, 1, 1]))

        # Harder case
        self.assertAlmostEqual(1.166860716, PropertyStats.geom_std_dev([0.5, 1.5, 1]))
        self.assertAlmostEqual(1.352205875, PropertyStats.geom_std_dev([0.5, 1.5, 1], weights=[2, 1, 0]))

    def test_quantile(self):
        self._run_test("quantile::0.5", 1, 1, 0.5, 0.5)
        self._run_test("quantile::0.3", 1, 1, 0.3, 0.3)
