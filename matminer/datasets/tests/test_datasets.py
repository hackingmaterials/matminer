import unittest
import os

import numpy as np
from pandas.api.types import is_numeric_dtype, is_object_dtype, is_bool_dtype
from pymatgen.core.structure import Structure, Composition

from matminer.datasets.tests import DataSetTest
from matminer.datasets.dataset_retrieval import load_dataset


class DataSetsTest(DataSetTest):
    def universal_dataset_check(self, dataset_name, object_headers=None,
                                numeric_headers=None, bool_headers=None):
        # Runs tests common to all datasets,
        # makes it quicker to write tests for new datasets

        # Get rid of dataset if it's on the disk already
        data_path = os.path.join(
            self.dataset_dir,
            dataset_name + "." + self.dataset_dict[dataset_name]['file_type']
        )
        if os.path.exists(data_path):
            os.remove(data_path)

        # Test that dataset can be downloaded
        load_dataset(dataset_name)
        self.assertTrue(os.path.exists(data_path))

        # Test that data is now available and has all its elements
        df = load_dataset(dataset_name, download_if_missing=False)
        self.assertEqual(
            len(df), self.dataset_dict[dataset_name]["num_entries"]
        )

        # Test all columns are there
        self.assertEqual(sorted(list(df)), sorted(
            [header for header in
             self.dataset_dict[dataset_name]['columns'].keys()]
        ))

        # Test each column for appropriate type
        if object_headers is None:
            object_headers = []
        if numeric_headers is None:
            numeric_headers = []
        if bool_headers is None:
            bool_headers = []

        df = load_dataset(dataset_name, download_if_missing=False)
        if object_headers:
            self.assertTrue(is_object_dtype(df[object_headers].values))
        if numeric_headers:
            self.assertTrue(is_numeric_dtype(df[numeric_headers].values))
        if bool_headers:
            self.assertTrue(is_bool_dtype(df[bool_headers].values))

        # Make sure all columns are accounted for
        column_headers = object_headers + numeric_headers + bool_headers
        self.assertEqual(sorted(list(df)), sorted(column_headers))

    # Skip for circleCI efficiency
    @unittest.skip
    def test_elastic_tensor_2015(self):
        # Run set of universal dataset tests
        object_headers = ['material_id', 'formula', 'structure',
                          'compliance_tensor', 'elastic_tensor',
                          'elastic_tensor_original', 'cif', 'poscar']

        numeric_headers = ['nsites', 'space_group', 'volume',
                           'elastic_anisotropy', 'G_Reuss', 'G_VRH', 'G_Voigt',
                           'K_Reuss', 'K_VRH', 'K_Voigt', 'poisson_ratio',
                           'kpoint_density']

        self.universal_dataset_check(
            "elastic_tensor_2015", object_headers, numeric_headers,
        )

        # Tests unique to this dataset
        df = load_dataset('elastic_tensor_2015')
        self.assertEqual(type(df['structure'][0]), Structure)
        tensor_headers = ['compliance_tensor', 'elastic_tensor',
                          'elastic_tensor_original']
        for c in tensor_headers:
            self.assertEqual(type(df[c][0]), np.ndarray)

    def test_piezoelectric_tensor(self):
        # Run universal tests
        object_headers = ['material_id', 'formula', 'structure', 'point_group',
                          'v_max', 'piezoelectric_tensor', 'cif', 'meta',
                          'poscar']

        numeric_headers = ['nsites', 'space_group', 'volume', 'eij_max']

        self.universal_dataset_check(
            "piezoelectric_tensor", object_headers, numeric_headers,
        )

        # Dataset specific checks
        df = load_dataset('piezoelectric_tensor')
        self.assertEqual(type(df['structure'][0]), Structure)
        self.assertEqual(type(df['piezoelectric_tensor'][0]), np.ndarray)

    def test_dielectric_constant(self):
        # Universal Tests
        object_headers = ['material_id', 'formula', 'structure',
                          'e_electronic', 'e_total', 'cif', 'meta',
                          'poscar']

        numeric_headers = ['nsites', 'space_group', 'volume', 'band_gap',
                           'n', 'poly_electronic', 'poly_total']

        bool_headers = ['pot_ferroelectric']

        self.universal_dataset_check(
            "dielectric_constant", object_headers, numeric_headers,
            bool_headers=bool_headers,
        )

        # Unique tests
        df = load_dataset("dielectric_constant")
        self.assertEqual(type(df['structure'][0]), Structure)

    # Skip for circleCI efficiency
    @unittest.skip
    def test_flla(self):
        # Universal tests
        object_headers = ['material_id', 'formula', 'structure']

        numeric_headers = ['e_above_hull', 'nsites', 'formation_energy',
                           'formation_energy_per_atom']

        self.universal_dataset_check("flla", object_headers, numeric_headers)

        # Unique tests
        df = load_dataset('flla')
        self.assertEqual(type(df['structure'][0]), Structure)

    # Skip for circleCI efficiency
    @unittest.skip
    def test_castelli_perovskites(self):
        # Universal Tests
        object_headers = ['structure', 'formula']

        numeric_headers = ['fermi level', 'fermi width', 'e_form', 'mu_b',
                           'vbm', 'cbm', 'gap gllbsc']

        bool_headers = ['gap is direct']

        self.universal_dataset_check(
            "castelli_perovskites", object_headers, numeric_headers,
            bool_headers=bool_headers
        )

        # Unique tests
        df = load_dataset("castelli_perovskites")
        self.assertEqual(type(df['structure'][0]), Structure)

    # Skip for circleCI efficiency
    @unittest.skip
    def test_boltztrap_mp(self):
        # Universal Tests
        object_headers = ['structure', 'formula', 'mpid']

        numeric_headers = ['pf_n', 'pf_p', 's_n', 's_p',
                           'm_n', 'm_p']

        self.universal_dataset_check(
            "boltztrap_mp", object_headers, numeric_headers,
        )

        # Unique tests
        df = load_dataset("boltztrap_mp")
        self.assertEqual(type(df['structure'][0]), Structure)

    # Skip for circleCI efficiency
    @unittest.skip
    def test_phonon_dielectric_mp(self):
        # Universal Tests
        object_headers = ['structure', 'formula', 'mpid']

        numeric_headers = ['eps_electronic', 'eps_total', 'last phdos peak']

        self.universal_dataset_check(
            "phonon_dielectric_mp", object_headers, numeric_headers,
        )

        # Unique tests
        df = load_dataset("phonon_dielectric_mp")
        self.assertEqual(type(df['structure'][0]), Structure)

    def test_glass_ternary_hipt(self):
        # Universal Tests
        object_headers = ['formula', 'system', 'processing', 'phase']

        numeric_headers = ['gfa']

        self.universal_dataset_check(
            "glass_ternary_hipt", object_headers, numeric_headers,
        )

    def test_double_perovskites_gap(self):
        # Universal Tests
        object_headers = ['formula', 'a_1', 'b_1', 'a_2', 'b_2']

        numeric_headers = ['gap gllbsc']

        self.universal_dataset_check(
            "double_perovskites_gap", object_headers, numeric_headers,
        )

    def test_double_perovskites_gap_lumo(self):
        # Universal Tests
        object_headers = ['atom']

        numeric_headers = ['lumo']

        self.universal_dataset_check(
            "double_perovskites_gap_lumo", object_headers, numeric_headers,
        )

    # Skip due to memory usage
    @unittest.skip
    def test_mp_all(self):
        # Universal Tests
        object_headers = ['mpid', 'formula', 'structure', 'initial structure']

        numeric_headers = ['e_hull', 'gap pbe', 'mu_b', 'elastic anisotropy',
                           'bulk modulus', 'shear modulus', 'e_form']

        self.universal_dataset_check(
            "mp_all", object_headers, numeric_headers,
        )

        # Unique tests
        df = load_dataset("mp_all")
        self.assertEqual(type(df['structure'][0]), Structure)

    # Skip for circleCI efficiency
    @unittest.skip
    def test_mp_nostruct(self):
        # Universal Tests
        object_headers = ['mpid', 'formula']

        numeric_headers = ['e_hull', 'gap pbe', 'mu_b', 'elastic anisotropy',
                           'bulk modulus', 'shear modulus', 'e_form']

        self.universal_dataset_check(
            "mp_nostruct", object_headers, numeric_headers,
        )

    def test_glass_ternary_landolt(self):
        # Universal Tests
        object_headers = ['phase', 'formula', 'processing']

        numeric_headers = ['gfa']

        self.universal_dataset_check(
            "glass_ternary_landolt", object_headers, numeric_headers,
        )

    def test_citrine_thermal_conductivity(self):
        # Universal Tests
        object_headers = ['k-units', 'formula', 'k_condition',
                          'k_condition_units']

        numeric_headers = ['k_expt']

        self.universal_dataset_check(
            "citrine_thermal_conductivity", object_headers, numeric_headers
        )

    def test_wolverton_oxides(self):
        # Universal Tests
        object_headers = ['atom a', 'formula', 'atom b', 'lowest distortion',
                          'mu_b', 'a', 'b', 'c', 'alpha', 'beta', 'gamma']

        numeric_headers = ['e_form', 'e_hull', 'vpa', 'gap pbe',
                           'e_form oxygen']

        self.universal_dataset_check(
            "wolverton_oxides", object_headers, numeric_headers
        )

    def test_heusler_magnetic(self):
        # Universal Tests
        object_headers = ['formula', 'heusler type', 'struct type']

        numeric_headers = ['num_electron', 'latt const', 'tetragonality',
                           'e_form', 'pol fermi', 'mu_b', 'mu_b saturation']

        self.universal_dataset_check(
            "heusler_magnetic", object_headers, numeric_headers
        )

    def test_steel_strength(self):
        # Universal Tests
        object_headers = ['formula']

        numeric_headers = ['c', 'mn', 'si', 'cr', 'ni', 'mo', 'v', 'n', 'nb',
                           'co', 'w', 'al', 'ti', 'yield strength',
                           'tensile strength', 'elongation']

        self.universal_dataset_check(
            "steel_strength", object_headers, numeric_headers
        )

    # Skip for circleCI efficiency
    @unittest.skip
    def test_jarvis_ml_dft_training(self):
        # Universal Tests
        object_headers = ['jid', 'mpid', 'structure', 'composition']

        numeric_headers = ['e mass_x', 'e mass_y', 'e mass_z',
                           'epsilon_x opt', 'epsilon_y opt', 'epsilon_z opt',
                           'e_exfol', 'e_form', 'shear modulus', 'hole mass_x',
                           'hole mass_y', 'hole mass_z', 'bulk modulus',
                           'mu_b', 'gap tbmbj', 'epsilon_x tbmbj',
                           'epsilon_y tbmbj', 'epsilon_z tbmbj', 'gap opt']

        self.universal_dataset_check(
            "jarvis_ml_dft_training", object_headers, numeric_headers,
        )

        # Unique tests
        df = load_dataset("jarvis_ml_dft_training")
        self.assertEqual(type(df['structure'][0]), Structure)
        self.assertEqual(type(df['composition'][0]), Composition)

    # Skip for circleCI efficiency
    @unittest.skip
    def test_jarvis_dft_3d(self):
        # Universal Tests
        object_headers = ['jid', 'mpid', 'structure', 'composition',
                          'structure initial']

        numeric_headers = ['epsilon_x opt', 'epsilon_y opt', 'epsilon_z opt',
                           'e_form', 'shear modulus', 'bulk modulus',
                           'gap tbmbj', 'epsilon_x tbmbj',
                           'epsilon_y tbmbj', 'epsilon_z tbmbj', 'gap opt']

        self.universal_dataset_check(
            "jarvis_dft_3d", object_headers, numeric_headers,
        )

        # Unique tests
        df = load_dataset("jarvis_dft_3d")
        self.assertEqual(type(df['structure'][0]), Structure)
        self.assertEqual(type(df['composition'][0]), Composition)

    # Skip for circleCI efficiency
    @unittest.skip
    def test_jarvis_dft_2d(self):
        # Universal Tests
        object_headers = ['jid', 'mpid', 'structure', 'composition',
                          'structure initial']

        numeric_headers = ['epsilon_x opt', 'epsilon_y opt', 'epsilon_z opt',
                           'exfoliation_en', 'e_form', 'gap tbmbj',
                           'epsilon_x tbmbj', 'epsilon_y tbmbj',
                           'epsilon_z tbmbj', 'gap opt']

        self.universal_dataset_check(
            "jarvis_dft_2d", object_headers, numeric_headers,
        )

        # Unique tests
        df = load_dataset("jarvis_dft_2d")
        self.assertEqual(type(df['structure'][0]), Structure)
        self.assertEqual(type(df['composition'][0]), Composition)

    def test_glass_binary(self):
        # Universal Tests
        object_headers = ['formula']

        numeric_headers = ['gfa']

        self.universal_dataset_check(
            "glass_binary", object_headers, numeric_headers
        )

    def test_m2ax(self):
        # Universal Tests
        object_headers = ['formula']

        numeric_headers = ['a', 'c', 'd_mx', 'd_ma', 'c11', 'c12', 'c13',
                           'c33', 'c44', 'bulk modulus', 'shear modulus',
                           'elastic modulus']

        self.universal_dataset_check(
            "m2ax", object_headers, numeric_headers
        )

    def test_expt_gap(self):
        # Universal Tests
        object_headers = ['formula']

        numeric_headers = ['gap expt']

        self.universal_dataset_check(
            "expt_gap", object_headers, numeric_headers
        )

    def test_expt_formation_enthalpy(self):
        # Universal Tests
        object_headers = ['formula', 'pearson symbol', 'space group', 'mpid']

        numeric_headers = ['oqmdid', 'e_form expt', 'e_form mp', 'e_form oqmd']

        self.universal_dataset_check(
            "expt_formation_enthalpy", object_headers, numeric_headers
        )


if __name__ == "__main__":
    unittest.main()
