import unittest
import os

import numpy as np
from pandas.api.types import is_numeric_dtype, is_object_dtype, is_bool_dtype
from pymatgen.core.structure import Structure

from matminer.datasets.tests import DataSetTest
from matminer.datasets.dataset_retrieval import load_dataset


class DataSetsTest(DataSetTest):
    def test_elastic_tensor_2015(self):
        # Test that the dataset is downloadable, also get integrity check
        # from internal check against file hash
        data_path = os.path.join(self.dataset_dir,
                                 "elastic_tensor_2015.json.gz")
        if os.path.exists(data_path):
            os.remove(data_path)

        load_dataset('elastic_tensor_2015')
        self.assertTrue(os.path.exists(data_path))

        # Test that data is now available and properly formatted
        df = load_dataset('elastic_tensor_2015', download_if_missing=False)
        self.assertEqual(type(df['structure'][0]), Structure)
        tensor_headers = ['compliance_tensor', 'elastic_tensor',
                          'elastic_tensor_original']
        for c in tensor_headers:
            self.assertEqual(type(df[c][0]), np.ndarray)
        self.assertEqual(len(df), 1181)

        object_headers = ['material_id', 'formula', 'structure',
                          'compliance_tensor', 'elastic_tensor',
                          'elastic_tensor_original', 'cif', 'poscar']

        numeric_headers = ['nsites', 'space_group', 'volume',
                           'elastic_anisotropy', 'G_Reuss', 'G_VRH', 'G_Voigt',
                           'K_Reuss', 'K_VRH', 'K_Voigt', 'poisson_ratio',
                           'kpoint_density']

        metadata_headers = {'cif', 'kpoint_density', 'poscar'}
        column_headers = object_headers + numeric_headers

        self.assertEqual(sorted(list(df)), sorted(
            [header for header in column_headers
             if header not in metadata_headers]))
        df = load_dataset('elastic_tensor_2015', include_metadata=True,
                          download_if_missing=False)
        self.assertEqual(sorted(list(df)), sorted(column_headers))
        # Test that each column is the right type
        self.assertTrue(is_object_dtype(df[object_headers].dtypes))
        self.assertTrue(is_numeric_dtype(df[numeric_headers].dtypes))

        os.remove(data_path)

    def test_piezoelectric_tensor(self):
        # Test that the dataset is downloadable, also get integrity check
        # from internal check against file hash
        data_path = os.path.join(self.dataset_dir,
                                 "piezoelectric_tensor.json.gz")
        if os.path.exists(data_path):
            os.remove(data_path)

        load_dataset('piezoelectric_tensor')
        self.assertTrue(os.path.exists(data_path))

        # Test that data is now available and properly formatted
        df = load_dataset("piezoelectric_tensor", download_if_missing=False)
        self.assertEqual(type(df['piezoelectric_tensor'][0]), np.ndarray)
        self.assertEqual(type(df['structure'][0]), Structure)
        self.assertEqual(len(df), 941)

        object_headers = ['material_id', 'formula', 'structure', 'point_group',
                          'v_max', 'piezoelectric_tensor', 'cif', 'meta',
                          'poscar']

        numeric_headers = ['nsites', 'space_group', 'volume', 'eij_max']

        metadata_headers = {'cif', 'meta', 'poscar'}
        column_headers = object_headers + numeric_headers

        self.assertEqual(sorted(list(df)), sorted(
            [header for header in column_headers
             if header not in metadata_headers]
        ))
        df = load_dataset('piezoelectric_tensor', include_metadata=True,
                          download_if_missing=False)
        self.assertEqual(sorted(list(df)), sorted(column_headers))
        # Test that each column is the right type
        self.assertTrue(is_object_dtype(df[object_headers].dtypes))
        self.assertTrue(is_numeric_dtype(df[numeric_headers].dtypes))

        os.remove(data_path)

    def test_dielectric_constant(self):
        # Test that the dataset is downloadable, also get integrity check
        # from internal check against file hash
        data_path = os.path.join(self.dataset_dir,
                                 "dielectric_constant.json.gz")
        if os.path.exists(data_path):
            os.remove(data_path)

        load_dataset('dielectric_constant')
        self.assertTrue(os.path.exists(data_path))

        # Test that data is now available and properly formatted
        df = load_dataset("dielectric_constant", download_if_missing=False)
        self.assertEqual(type(df['structure'][0]), Structure)
        self.assertEqual(len(df), 1056)

        object_headers = ['material_id', 'formula', 'structure',
                          'e_electronic', 'e_total', 'cif', 'meta',
                          'poscar']

        numeric_headers = ['nsites', 'space_group', 'volume', 'band_gap',
                           'n', 'poly_electronic', 'poly_total']

        bool_headers = ['pot_ferroelectric']

        metadata_headers = {'cif', 'meta', 'poscar'}

        column_headers = object_headers + numeric_headers + bool_headers
        self.assertEqual(sorted(list(df)), sorted(
            [header for header in column_headers
             if header not in metadata_headers]
        ))
        df = load_dataset("dielectric_constant", include_metadata=True,
                          download_if_missing=False)
        self.assertEqual(sorted(list(df)), sorted(column_headers))
        # Test that each column is the right type
        self.assertTrue(is_object_dtype(df[object_headers].dtypes))
        self.assertTrue(is_numeric_dtype(df[numeric_headers].dtypes))
        self.assertTrue(is_bool_dtype(df[bool_headers].dtypes))

        os.remove(data_path)

    def test_flla(self):
        # Test that the dataset is downloadable, also get integrity check
        # from internal check against file hash
        data_path = os.path.join(self.dataset_dir, "flla.json.gz")
        if os.path.exists(data_path):
            os.remove(data_path)

        load_dataset("flla")
        self.assertTrue(os.path.exists(data_path))

        # Test that data is now available and properly formatted
        df = load_dataset("flla", download_if_missing=False)
        self.assertEqual(type(df['structure'][0]), Structure)
        self.assertEqual(len(df), 3938)

        object_headers = ['material_id', 'formula', 'structure']

        numeric_headers = ['e_above_hull', 'nsites', 'formation_energy',
                           'formation_energy_per_atom']

        column_headers = object_headers + numeric_headers

        self.assertEqual(sorted(list(df)), sorted(column_headers))
        df = load_dataset('flla', include_metadata=True,
                          download_if_missing=False)
        self.assertEqual(sorted(list(df)), sorted(column_headers))
        # Test that each column is the right type
        self.assertTrue(is_object_dtype(df[object_headers].dtypes))
        self.assertTrue(is_numeric_dtype(df[numeric_headers].dtypes))

        os.remove(data_path)


if __name__ == "__main__":
    unittest.main()
