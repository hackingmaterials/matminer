import unittest
from pymatgen import MPRester
from matminer.learners.volume_prediction import VolumePredictor

mpr = MPRester()


class TestVolumePredictor(unittest.TestCase):
    def test_Si(self):
        s = mpr.get_structure_by_material_id('mp-149')
        self.assertAlmostEqual(VolumePredictor().predict(s), 32.8, 1)

    def test_CsF(self):
        s = mpr.get_structure_by_material_id('mp-1784')
        self.assertAlmostEqual(VolumePredictor().predict(s), 62.1, 1)

    def test_CsF_ionic(self):
        s = mpr.get_structure_by_material_id('mp-1784')
        self.assertAlmostEqual(VolumePredictor(ionic_factor=0.20).predict(s), 73.2, 1)

if __name__ == '__main__':
    unittest.main()
