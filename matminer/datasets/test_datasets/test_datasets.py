import unittest
import os
from itertools import product

import numpy as np
from pymatgen.core.structure import Structure

from matminer.datasets.dataframe_loader import _load_dataset_dict, \
    _get_data_home, _fetch_external_dataset, _validate_dataset, \
    load_dataset, available_datasets


class DataSetTest(unittest.TestCase):
    def setUp(self):
        self.dataset_names = [
            'flla',
            'elastic_tensor_2015',
            'piezoelectric_tensor',
            'dielectric_constant'
        ]
        self.dataset_attributes = [
            'file_name',
            'url',
            'hash',
            'reference',
            'description',
            'columns',
            'bibtex_refs',
            'num_entries'
        ]
        # current directory, for storing and discarding test_dataset
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # directory where in-use datasets should be stored,
        # either at MATMINER_DATA env var or under matminer/datasets/
        self.dataset_dir = os.environ.get(
            "MATMINER_DATA",
            os.path.abspath(os.path.join(current_dir, os.pardir))
        )

        # Shared set up for test_validate_dataset & test_fetch_external_dataset
        self._path = os.path.join(current_dir, "test_dataset.csv")
        self._url = "https://ndownloader.figshare.com/files/13039562"
        self._hash = "c487f59ce0d48505c36633b4b202027d0c915474b081e8fb0bde8d5474ee59a1"

    def test_load_dataset_dict(self):
        dataset_dict = _load_dataset_dict()
        # Check to make sure all datasets are present and have string type keys
        self.assertEqual(set(dataset_dict.keys()), set(self.dataset_names))
        # Check the validity of each set of values in each dataset
        for value in dataset_dict.values():
            # Check to make sure each dataset has all attributes
            # and string type keys
            self.assertEqual(set(value.keys()), set(self.dataset_attributes))
            # Make sure string attributes have string values
            for item in ['file_name', 'url', 'hash', 'reference', 'description',
                         'bibtex_refs']:
                self.assertIsInstance(value[item], str)
            # Make sure int attributes have int values
            self.assertIsInstance(value['num_entries'], int)
            # Make sure columns is a dict and it has string valued entries
            self.assertIsInstance(value['columns'], dict)
            for column_name, column_description in value['columns'].items():
                self.assertIsInstance(column_name, str)
                self.assertIsInstance(column_description, str)

    # This test case only checks the dataset loaders exceptions, for
    # tests of individual datasets see the test_load_"dataset name" functions
    def test_load_dataset(self):
        # Can't find dataset or similar
        with self.assertRaises(ValueError):
            load_dataset("not_real_dataset")
        # Finds similar
        with self.assertRaises(ValueError):
            load_dataset("tensor")
        # Actual dataset is subset of passed dataset name
        with self.assertRaises(ValueError):
            load_dataset('ffllaa')

        # Pick a dataset from the dictionary to try to load in different places
        dataset_info = _load_dataset_dict()
        chosen_dataset = list(dataset_info.keys())[0]

        data_home = os.path.expanduser("~")
        dataset_path = os.path.join(data_home,
                                    dataset_info[chosen_dataset]['file_name'])
        if os.path.exists(dataset_path):
            os.remove(dataset_path)

        load_dataset(chosen_dataset, data_home)
        self.assertTrue(os.path.exists(data_home))

    def test_available_datasets(self):
        # Get dataset info for checking sorting works properly
        dataset_dict = _load_dataset_dict()
        # Go over all parameter combinations,
        # for each check that returned dataset is correct
        for parameter_combo in product([True, False], [True, False],
                                       ['alphabetical', 'num_entries']):
            datasets = available_datasets(*parameter_combo)
            if parameter_combo[2] == 'alphabetical':
                self.assertEqual(datasets, sorted(self.dataset_names))
            else:
                self.assertEqual(
                    datasets,
                    sorted(self.dataset_names,
                           key=lambda x: dataset_dict[x]['num_entries'])
                )

    def test_get_data_home(self):
        home = _get_data_home()
        self.assertEqual(home, self.dataset_dir)
        specified_home = _get_data_home('/some/specified/path')
        self.assertEqual(specified_home, '/some/specified/path')

    def test_validate_dataset(self):
        if os.path.exists(self._path):
            os.remove(self._path)

        with self.assertRaises(IOError):
            _validate_dataset(self._path, self._url, self._hash,
                              download_if_missing=False)

        # Check to make sure the IOError takes precedence over the ValueError
        with self.assertRaises(IOError):
            _validate_dataset(self._path, url=None, file_hash=self._hash,
                              download_if_missing=False)

        with self.assertRaises(ValueError):
            _validate_dataset(self._path, url=None, file_hash=self._hash,
                              download_if_missing=True)

        with self.assertRaises(UserWarning):
            _validate_dataset(self._path, self._url, file_hash="!@#$%^&*",
                              download_if_missing=True)
        os.remove(self._path)

        _validate_dataset(self._path, self._url, self._hash,
                          download_if_missing=True)
        self.assertTrue(os.path.exists(self._path))
        os.remove(self._path)

        _validate_dataset(self._path, self._url, file_hash=None,
                          download_if_missing=True)
        self.assertTrue(os.path.exists(self._path))
        os.remove(self._path)

    def test_fetch_external_dataset(self):
        if os.path.exists(self._path):
            os.remove(self._path)

        _fetch_external_dataset(self._url, self._path)
        self.assertTrue(os.path.exists(self._path))
        os.remove(self._path)

    def test_elastic_tensor_2015(self):
        # Test that the dataset is downloadable, also get integrity check
        # from internal check against file hash
        data_path = os.path.join(self.dataset_dir, "elastic_tensor.csv")
        if os.path.exists(data_path):
            os.remove(data_path)

        load_dataset('elastic_tensor_2015')
        self.assertTrue(os.path.exists(data_path))

        # Test that data is now available and properly formatted
        df = load_dataset('elastic_tensor_2015', download_if_missing=False)
        self.assertEqual(type(df['structure'][0]), Structure)
        for c in ['compliance_tensor', 'elastic_tensor', 'elastic_tensor_original']:
            self.assertEqual(type(df[c][0]), np.ndarray)
        self.assertEqual(len(df), 1181)
        column_headers = ['material_id', 'formula', 'nsites', 'space_group',
                          'volume', 'structure', 'elastic_anisotropy',
                          'G_Reuss', 'G_VRH', 'G_Voigt', 'K_Reuss', 'K_VRH',
                          'K_Voigt', 'poisson_ratio', 'compliance_tensor',
                          'elastic_tensor', 'elastic_tensor_original']
        self.assertEqual(list(df), column_headers)
        df = load_dataset('elastic_tensor_2015', include_metadata=True,
                          download_if_missing=False)
        column_headers += ['cif', 'kpoint_density', 'poscar']
        self.assertEqual(list(df), column_headers)

        os.remove(data_path)

    def test_piezoelectric_tensor(self):
        # Test that the dataset is downloadable, also get integrity check
        # from internal check against file hash
        data_path = os.path.join(self.dataset_dir, "piezoelectric_tensor.csv")
        if os.path.exists(data_path):
            os.remove(data_path)

        load_dataset('piezoelectric_tensor')
        self.assertTrue(os.path.exists(data_path))

        # Test that data is now available and properly formatted
        df = load_dataset("piezoelectric_tensor", download_if_missing=False)
        self.assertEqual(len(df), 941)
        self.assertEqual(type(df['piezoelectric_tensor'][0]), np.ndarray)
        self.assertEqual(type(df['structure'][0]), Structure)
        column_headers = ['material_id', 'formula', 'nsites', 'point_group',
                          'space_group', 'volume', 'structure', 'eij_max',
                          'v_max', 'piezoelectric_tensor']
        self.assertEqual(list(df), column_headers)
        df = load_dataset("piezoelectric_tensor", include_metadata=True,
                          download_if_missing=False)
        column_headers += ['cif', 'meta', 'poscar']
        self.assertEqual(list(df), column_headers)

        os.remove(data_path)

    def test_dielectric_tensor(self):
        # Test that the dataset is downloadable, also get integrity check
        # from internal check against file hash
        data_path = os.path.join(self.dataset_dir, "dielectric_constant.csv")
        if os.path.exists(data_path):
            os.remove(data_path)

        load_dataset('dielectric_constant')
        self.assertTrue(os.path.exists(data_path))

        # Test that data is now available and properly formatted
        df = load_dataset("dielectric_constant", download_if_missing=False)
        self.assertEqual(type(df['structure'][0]), Structure)
        self.assertEqual(len(df), 1056)
        column_headers = ['material_id', 'formula', 'nsites', 'space_group',
                          'volume', 'structure', 'band_gap', 'e_electronic',
                          'e_total', 'n', 'poly_electronic', 'poly_total',
                          'pot_ferroelectric']
        self.assertEqual(list(df), column_headers)
        df = load_dataset("dielectric_constant", include_metadata=True,
                          download_if_missing=False)
        column_headers += ['cif', 'meta', 'poscar']
        self.assertEqual(list(df), column_headers)

        os.remove(data_path)

    def test_flla(self):
        # Test that the dataset is downloadable, also get integrity check
        # from internal check against file hash
        data_path = os.path.join(self.dataset_dir, "flla_2015.csv")
        if os.path.exists(data_path):
            os.remove(data_path)

        load_dataset("flla")
        self.assertTrue(os.path.exists(data_path))

        # Test that data is now available and properly formatted
        df = load_dataset("flla", download_if_missing=False)
        self.assertEqual(type(df['structure'][0]), Structure)
        self.assertEqual(len(df), 3938)
        column_headers = ['material_id', 'e_above_hull', 'formula',
                          'nsites', 'structure', 'formation_energy',
                          'formation_energy_per_atom']
        self.assertEqual(list(df), column_headers)

        os.remove(data_path)


if __name__ == "__main__":
    unittest.main()
