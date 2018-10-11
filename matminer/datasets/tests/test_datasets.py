import unittest
import os

import numpy as np
from pandas.api.types import is_numeric_dtype, is_object_dtype, is_bool_dtype
from pymatgen.core.structure import Structure

from matminer.datasets.tests import DataSetTest
from matminer.datasets.dataset_retrieval import load_dataset


class DataSetsTest(DataSetTest):
    def universal_dataset_test(self, dataset_name, object_headers=None,
                               numeric_headers=None, bool_headers=None,
                               metadata_headers=None):
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

        # Tes all the non-metadata columns are there
        if metadata_headers is None:
            metadata_headers = set()

        self.assertEqual(sorted(list(df)), sorted(
            [header for header in
             self.dataset_dict[dataset_name]['columns'].keys()
             if header not in metadata_headers]
        ))

        # Test each column for appropriate type and
        if object_headers is None:
            object_headers = []
        if numeric_headers is None:
            numeric_headers = []
        if bool_headers is None:
            bool_headers = []

        df = load_dataset(dataset_name, include_metadata=True,
                          download_if_missing=False)
        if object_headers:
            self.assertTrue(is_object_dtype(df[object_headers].values))
        if numeric_headers:
            self.assertTrue(is_numeric_dtype(df[numeric_headers].values))
        if bool_headers:
            self.assertTrue(is_bool_dtype(df[bool_headers].values))

        # Make sure all columns are accounted for
        column_headers = object_headers + numeric_headers + bool_headers
        self.assertEqual(sorted(list(df)), sorted(column_headers))

    def test_elastic_tensor_2015(self):
        # Run set of universal dataset tests
        object_headers = ['material_id', 'formula', 'structure',
                          'compliance_tensor', 'elastic_tensor',
                          'elastic_tensor_original', 'cif', 'poscar']

        numeric_headers = ['nsites', 'space_group', 'volume',
                           'elastic_anisotropy', 'G_Reuss', 'G_VRH', 'G_Voigt',
                           'K_Reuss', 'K_VRH', 'K_Voigt', 'poisson_ratio',
                           'kpoint_density']

        metadata_headers = {'cif', 'kpoint_density', 'poscar'}

        self.universal_dataset_test(
            "elastic_tensor_2015", object_headers, numeric_headers,
            metadata_headers=metadata_headers
        )

        # Tests unique to this dataset
        df = load_dataset('elastic_tensor_2015', include_metadata=True,
                          download_if_missing=False)
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

        metadata_headers = {'cif', 'meta', 'poscar'}

        self.universal_dataset_test(
            "piezoelectric_tensor", object_headers, numeric_headers,
            metadata_headers=metadata_headers
        )

        # Dataset specific checks
        df = load_dataset('piezoelectric_tensor', include_metadata=True,
                          download_if_missing=False)
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

        metadata_headers = {'cif', 'meta', 'poscar'}

        self.universal_dataset_test(
            "dielectric_constant", object_headers, numeric_headers,
            bool_headers=bool_headers, metadata_headers=metadata_headers
        )

        # Unique tests
        df = load_dataset("dielectric_constant", include_metadata=True,
                          download_if_missing=False)
        self.assertEqual(type(df['structure'][0]), Structure)

    def test_flla(self):
        # Universal tests
        object_headers = ['material_id', 'formula', 'structure']

        numeric_headers = ['e_above_hull', 'nsites', 'formation_energy',
                           'formation_energy_per_atom']

        self.universal_dataset_test("flla", object_headers, numeric_headers)

        # Unique tests
        df = load_dataset('flla', include_metadata=True,
                          download_if_missing=False)
        self.assertEqual(type(df['structure'][0]), Structure)


if __name__ == "__main__":
    unittest.main()
