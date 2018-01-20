from __future__ import unicode_literals, division, print_function

import unittest
import pandas as pd
import numpy as np

from pymatgen.util.testing import PymatgenTest

from matminer.featurizers.base import BaseFeaturizer, MultipleFeaturizer


class SingleFeaturizer(BaseFeaturizer):
    def feature_labels(self):
        return ['y']

    def featurize(self, x):
        return [x + 1]

    def citations(self):
        return ["A"]

    def implementors(self):
        return ["Us"]


class MultipleFeatureFeaturizer(BaseFeaturizer):
    def feature_labels(self):
        return ['w', 'z']

    def featurize(self, x):
        return [x - 1, x + 2]

    def citations(self):
        return ["A"]

    def implementors(self):
        return ["Them"]


class MatrixFeaturizer(BaseFeaturizer):
    def feature_labels(self):
        return ['representation']

    def featurize(self, *x):
        return [np.eye(2, 2)]

    def citations(self):
        return ["C"]

    def implementors(self):
        return ["Everyone"]


class TestBaseClass(PymatgenTest):
    def setUp(self):
        self.single = SingleFeaturizer()
        self.multi = MultipleFeatureFeaturizer()
        self.matrix = MatrixFeaturizer()

    @staticmethod
    def make_test_data():
        return pd.DataFrame({'x': [1, 2, 3]})

    def test_dataframe(self):
        data = self.make_test_data()
        data = self.single.featurize_dataframe(data, 'x')
        self.assertArrayAlmostEqual(data['y'], [2, 3, 4])

        data = self.multi.featurize_dataframe(data, 'x')
        self.assertArrayAlmostEqual(data['w'], [0, 1, 2])
        self.assertArrayAlmostEqual(data['z'], [3, 4, 5])

    def test_matrix(self):
        """Test the ability to add features that are matrices to a dataframe"""
        data = self.make_test_data()
        data = self.matrix.featurize_dataframe(data, 'x')
        self.assertArrayAlmostEqual(np.eye(2, 2), data['representation'][0])

    def test_inplace(self):
        data = self.make_test_data()

        self.single.featurize_dataframe(data, 'x', inplace=False)
        self.assertNotIn('y', data.columns)

        self.single.featurize_dataframe(data, 'x', inplace=True)
        self.assertIn('y', data)

    def test_indices(self):
        data = self.make_test_data()
        data.index = [4, 6, 6]

        data = self.single.featurize_dataframe(data, 'x')
        self.assertArrayAlmostEqual(data['y'], [2, 3, 4])

    def test_multiple(self):
        multi_f = MultipleFeaturizer([self.single, self.multi])
        data = self.make_test_data()

        self.assertArrayAlmostEqual([2, 0, 3], multi_f.featurize(1))

        self.assertArrayEqual(['A'], multi_f.citations())

        implementors = multi_f.implementors()
        self.assertIn('Us', implementors)
        self.assertIn('Them', implementors)
        self.assertEquals(2, len(implementors))

        multi_f.featurize_dataframe(data, 'x')
        self.assertArrayAlmostEqual(data['y'], [2, 3, 4])
        self.assertArrayAlmostEqual(data['w'], [0, 1, 2])
        self.assertArrayAlmostEqual(data['z'], [3, 4, 5])


if __name__ == '__main__':
    unittest.main()
