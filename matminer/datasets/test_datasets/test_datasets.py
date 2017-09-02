import unittest

from matminer.datasets.dataframe_loader import load_elastic_tensor, \
    load_piezoelectric_tensor, load_dielectric_constant


class DataSetTest(unittest.TestCase):
    def test_elastic_tensor(self):
        df = load_elastic_tensor()
        self.assertEqual(len(df), 1181)
        column_headers = {'idx', 'G_Reuss', 'G_VRH', 'G_Voigt', 'K_Reuss',
                          'K_VRH', 'K_Voigt', 'compliance_tensor',
                          'elastic_anisotropy', 'elastic_tensor',
                          'elastic_tensor_original', 'formula',
                          'kpoint_density', 'material_id', 'nsites',
                          'poisson_ratio', 'poscar', 'space_group', 'structure',
                          'volume'}
        self.assertEqual(set(list(df)), column_headers)

    def test_piezoelectric_tensor(self):
        df = load_piezoelectric_tensor()
        self.assertEqual(len(df), 941)
        column_headers = {'idx', 'eij_max', 'meta', 'piezoelectric_tensor',
                          'v_max', 'structure', 'poscar', 'formula', 'volume',
                          'space_group', 'point_group', 'nsites', 'material_id'}
        self.assertEqual(set(list(df)), column_headers)

    def test_dielectric_tensor(self):
        df = load_dielectric_constant()
        self.assertEqual(len(df), 1056)
        print(list(df))
        column_headers = {'idx', 'band_gap', 'e_electronic', 'e_total', 'meta',
                          'n', 'poly_electronic', 'poly_total',
                          'pot_ferroelectric', 'nsites', 'structure', 'poscar',
                          'space_group', 'material_id', 'formula', 'volume'}
        self.assertEqual(set(list(df)), column_headers)


if __name__ == "__main__":
    unittest.main()
