import unittest
import numpy as np
from pymatgen.core.structure import Structure

from matminer.datasets.dataframe_loader import load_elastic_tensor, \
    load_piezoelectric_tensor, load_dielectric_constant, load_flla


class DataSetTest(unittest.TestCase):
    def test_elastic_tensor(self):
        df = load_elastic_tensor()
        self.assertEqual(type(df['structure'][0]), Structure)
        for c in ['compliance_tensor', 'elastic_tensor', 'elastic_tensor_original']:
            self.assertEqual(type(df[c][0]), np.ndarray)
        self.assertEqual(len(df), 1181)
        column_headers = ['material_id', 'formula',
                          'nsites', 'space_group', 'volume',
                          'structure', 'elastic_anisotropy', 'G_Reuss',
                          'G_VRH', 'G_Voigt', 'K_Reuss',
                          'K_VRH', 'K_Voigt', 'poisson_ratio',
                          'compliance_tensor', 'elastic_tensor',
                          'elastic_tensor_original']
        self.assertEqual(list(df), column_headers)
        df = load_elastic_tensor(include_metadata = True)
        column_headers += ['cif', 'kpoint_density', 'poscar']
        self.assertEqual(list(df), column_headers)

    def test_piezoelectric_tensor(self):
        df = load_piezoelectric_tensor()
        self.assertEqual(len(df), 941)
        self.assertEqual(type(df['piezoelectric_tensor'][0]), np.ndarray)
        self.assertEqual(type(df['structure'][0]), Structure)
        column_headers = ['material_id', 'formula',
                          'nsites', 'point_group', 'space_group', 'volume',
                          'structure', 'eij_max', 'v_max', 'piezoelectric_tensor']
        self.assertEqual(list(df), column_headers)
        df = load_piezoelectric_tensor(include_metadata = True)
        column_headers += ['cif', 'meta', 'poscar']
        self.assertEqual(list(df), column_headers)

    def test_dielectric_tensor(self):
        df = load_dielectric_constant()
        self.assertEqual(type(df['structure'][0]), Structure)
        self.assertEqual(len(df), 1056)
        column_headers = ['material_id', 'formula',
                          'nsites', 'space_group', 'volume',
                          'structure',
                          'band_gap', 'e_electronic', 'e_total',
                          'n', 'poly_electronic',
                          'poly_total', 'pot_ferroelectric']
        self.assertEqual(list(df), column_headers)
        df = load_dielectric_constant(include_metadata = True)
        column_headers += ['cif', 'meta', 'poscar']
        self.assertEqual(list(df), column_headers)

    def test_flla(self):
        df = load_flla()
        self.assertEqual(type(df['structure'][0]), Structure)
        self.assertEqual(len(df), 3938)
        column_headers = ['material_id', 'e_above_hull', 'formula',
                          'nsites', 'structure', 'formation_energy',
                          'formation_energy_per_atom']
        self.assertEqual(list(df), column_headers)


if __name__ == "__main__":
    unittest.main()
