import unittest
from matminer.learners.volume_predictor import VolumePredictor
from pymatgen.util.testing import PymatgenTest


class TestVolumePredictor(unittest.TestCase):
    def test_Si(self):
        s = PymatgenTest.get_structure("Si")
        self.assertAlmostEqual(VolumePredictor().predict(s), 32.8, 1)

    def test_CsCl(self):
        s = PymatgenTest.get_structure("CsCl")
        self.assertAlmostEqual(VolumePredictor().predict(s), 99.78, 1)

    def test_CsCl_ionic(self):
        s = PymatgenTest.get_structure("CsCl")
        self.assertAlmostEqual(VolumePredictor(ionic_factor=0.20).predict(s),
                               112.36, 1)

if __name__ == '__main__':
    unittest.main()
