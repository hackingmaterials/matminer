from __future__ import unicode_literals, division, print_function

import unittest
import pandas as pd

from pymatgen.util.testing import PymatgenTest

from matminer.featurizers.base import BaseFeaturizer


class SingleFeaturizer(BaseFeaturizer):
    def feature_labels(self):
        return ['y']
       
    def featurize(self, x):
        return x + 1
        
class MultipleFeaturizer(BaseFeaturizer):
    def feature_labels(self):
        return ['y', 'z']
       
    def featurize(self, x):
        return [x - 1, x + 2]

class TestBaseClass(PymatgenTest):
    
    def setUp(self):
        self.single = SingleFeaturizer()
        self.multi = MultipleFeaturizer()
        
    def test_dataframe(self):
        data = pd.DataFrame({'x': [1,2,3]})
        data = self.single.featurize_dataframe(data, 'x')
        self.assertArrayAlmostEqual(data['y'], [2,3,4])
        
        data = self.multi.featurize_dataframe(data, 'x')
        self.assertArrayAlmostEqual(data['y'], [0,1,2])
        self.assertArrayAlmostEqual(data['z'], [3,4,5])
        
        
        

if __name__ == '__main__':
    unittest.main()
