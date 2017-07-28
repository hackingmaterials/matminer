from unittest import TestCase

from matminer.featurizers.stats import PropertyStats
import numpy as np
from math import sqrt


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

        self.assertAlmostEqual(sample_1, PropertyStats.calc_stat(statistic, self.sample_1))
        self.assertAlmostEqual(sample_1_weighted, PropertyStats.calc_stat(statistic, self.sample_1,
                                                                          self.sample_1_weights))
        self.assertAlmostEqual(sample_2, PropertyStats.calc_stat(statistic, self.sample_2))
        self.assertAlmostEqual(sample_2_weighted, PropertyStats.calc_stat(statistic, self.sample_2,
                                                                          self.sample_2_weights ))

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
        self._run_test("std_dev", 0, 0, 0.623609564, 0.524890659)

    def test_mode(self):
        self._run_test("mode", 1, 1, 0, 0.5)

        # Additional tests
        self.assertAlmostEqual(0, PropertyStats.mode([0,1,2], [1,1,1]))

    def test_holder_mean(self):
        # Tests an edge case where we a value of zero
        self._run_test("holder_mean__0", 1, 1, np.product(self.sample_2), 0)

        self._run_test("holder_mean__1", 1, 1, 2./3, 5./7)
        self._run_test("holder_mean__2", 1, 1, sqrt(5./6), 0.88640526)
