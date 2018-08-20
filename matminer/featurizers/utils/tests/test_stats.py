from __future__ import unicode_literals, division, print_function

from math import sqrt
from unittest import TestCase

import numpy as np

from matminer.featurizers.utils.stats import PropertyStats


class TestPropertyStats(TestCase):

    def setUp(self):
        self.sample_1 = [1]*3
        self.sample_1_weights = [1] * 3
        self.sample_2 = [0.5, 1.5, 0]
        self.sample_2_weights = [2, 1, 0.5]

    def _run_test(self, statistic, sample_1, sample_1_weighted, sample_2, sample_2_weighted):
        """ Run a test for a certain statistic against the two sample datasets

        :param statistic: name of statistic
        :param sample_1: float, expected value for statistic of sample 1 without weights
        :param sample_1_weighted: float, expected value for statistic of sample 1 with weights
        :param sample_2: float, expected value for statistic of sample 2 without weights
        :param sample_2_weighted: float, expected value for statistic of sample 2 with weights
        """

        self.assertAlmostEqual(sample_1, PropertyStats.calc_stat(self.sample_1, statistic))
        self.assertAlmostEqual(sample_1_weighted, PropertyStats.calc_stat(self.sample_1, statistic,
                                                                          self.sample_1_weights))
        self.assertAlmostEqual(sample_2, PropertyStats.calc_stat(self.sample_2, statistic))
        self.assertAlmostEqual(sample_2_weighted, PropertyStats.calc_stat(self.sample_2, statistic,
                                                                          self.sample_2_weights))

    def test_minimum(self):
        self._run_test("minimum", 1, 1, 0, 0)

    def test_maximum(self):
        self._run_test("maximum", 1, 1, 1.5, 1.5)

    def test_range(self):
        self._run_test("range", 0, 0, 1.5, 1.5)

    def test_mean(self):
        self._run_test("mean", 1, 1, 2./3, 5./7)

    def test_avg_dev(self):
        self._run_test("avg_dev", 0, 0, 5./9, 0.448979592)

    def test_std_dev(self):
        self._run_test("std_dev", 0, 0, 0.623609564, 0.694365075)

    def test_skewness(self):
        self._run_test("skewness", 0, 0, 0.38180177, 0.559451361)

    def test_kurtosis(self):
        self._run_test("kurtosis", 0, 0, 1.5, 1.9403292181)

    def test_mode(self):
        self._run_test("mode", 1, 1, 0, 0.5)

        # Additional tests
        self.assertAlmostEqual(0, PropertyStats.mode([0,1,2], [1,1,1]))

    def test_holder_mean(self):
        self._run_test("holder_mean::0", 1, 1, np.product(self.sample_2), 0)

        self._run_test("holder_mean::1", 1, 1, 2./3, 5./7)
        self._run_test("holder_mean::2", 1, 1, sqrt(5./6), 0.88640526)

        # can't use run_test since it uses a sample with zero, which is not
        # allowed for Holder mean with -1
        self.assertAlmostEqual(PropertyStats.holder_mean(
            [1, 1, 2], power=-1), 1.2, places=3)

        self.assertAlmostEqual(PropertyStats.holder_mean(
            [1, 2], [2, 1], power=-1), 1.2, places=3)

    def test_geom_std_dev(self):
        # This is right. Yes, a list without variation has a geom_std_dev of 1
        self.assertAlmostEqual(1, PropertyStats.geom_std_dev([1, 1, 1]))

        # Harder case
        self.assertAlmostEqual(1.166860716, PropertyStats.geom_std_dev([0.5, 1.5, 1]))
        self.assertAlmostEqual(1.352205875, PropertyStats.geom_std_dev([0.5, 1.5, 1], weights=[2, 1, 0]))
