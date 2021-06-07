from pymatgen.util.testing import PymatgenTest
from pymatgen.core.composition import Composition, Species

from matminer.featurizers.utils.oxidation import has_oxidation_states


class OxidationTest(PymatgenTest):
    def test_has_oxidation_states(self):
        c_el = Composition("Fe2O3")
        c_sp = Composition({Species("Fe", 3): 2, Species("O", -2): 3})
        self.assertFalse(has_oxidation_states(c_el))
        self.assertTrue(has_oxidation_states(c_sp))
