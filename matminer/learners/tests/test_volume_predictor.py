import unittest
from matminer.learners.volume_predictor import VolumePredictor, ConditionalVolumePredictor
from pymatgen.util.testing import PymatgenTest


class VolumePredictorTest(unittest.TestCase):
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
        s.replace_species({"Cs": "Li", "Cl": "F"})
        self.assertAlmostEqual(VolumePredictor(ionic_factor=0.20).predict(s),
                               16.974592999999995, 1)


class ConditionalVolumePredictorTest(PymatgenTest):

    def test_predict(self):
        s = PymatgenTest.get_structure("CsCl")
        nacl = PymatgenTest.get_structure("CsCl")
        nacl.replace_species({"Cs": "Na"})
        nacl.scale_lattice(184.384551033)
        p = ConditionalVolumePredictor()
        self.assertAlmostEqual(p.predict(s, nacl), 342.84905395082535)
        lif = PymatgenTest.get_structure("CsCl")
        lif.replace_species({"Cs": "Li", "Cl": "F"})
        self.assertAlmostEqual(p.predict(lif, nacl), 74.268402413690467)

        lfpo = PymatgenTest.get_structure("LiFePO4")
        lmpo = PymatgenTest.get_structure("LiFePO4")
        lmpo.replace_species({"Fe": "Mn"})
        self.assertAlmostEqual(p.predict(lmpo, lfpo), 310.08253254420134)

        sto = PymatgenTest.get_structure("SrTiO3")
        scoo = PymatgenTest.get_structure("SrTiO3")
        scoo.replace_species({"Ti4+": "Co4+"})
        self.assertAlmostEqual(p.predict(scoo, sto), 56.162534974936463)

if __name__ == '__main__':
    unittest.main()
