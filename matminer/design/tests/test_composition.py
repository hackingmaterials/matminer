import unittest

from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element

from matminer.design.composition import CompositionGenerator


class TestCompositionGenerators(unittest.TestCase):

    def test_simple_generator(self):
        gen = CompositionGenerator(['Al', Element('Fe'), 'Zr'],
                                   min_elements=1, max_elements=2, spacing=4)

        # Test the stoichiometry generator
        stoichs = list(gen._generate_stoichiometries(1, 4))
        self.assertEquals([(4,)], stoichs)
        stoichs = set(gen._generate_stoichiometries(2, 4))
        self.assertEquals({(3, 1), (2, 2), (1, 3)}, stoichs)

        # Test the composition generator
        comps = list(gen.generate_entries())
        self.assertEquals(len(comps), len(set(comps)))  # No duplicates
        self.assertEquals(3 + 3 * 3, len(comps))  # 3 elements, 3 entries/binary * 3 binaries
        self.assertIn(Composition('Al4'), comps)
        self.assertIn(Composition('Al2Fe2'), comps)

        # Test the dataframe generator (from the base class)
        data = gen.generate_dataframe('composition')
        self.assertEquals(comps, data['composition'].tolist())

if __name__ == '__main__':
    unittest.main()
