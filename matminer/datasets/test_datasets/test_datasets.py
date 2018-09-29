import unittest
import os
import numpy as np
from pymatgen.core.structure import Structure

from matminer.datasets.dataframe_loader import load_elastic_tensor, \
    load_piezoelectric_tensor, load_dielectric_constant, load_flla, \
    RemoteFileMetadata, fetch_external_dataset, validate_dataset


class DataSetTest(unittest.TestCase):
    def setUp(self):
        # current directory, for storing and discarding test_dataset
        self.current_dir = os.path.dirname(os.path.abspath(__file__))

        # directory where in-use datasets should be stored,
        # either at MATMINER_DATA env var or under matminer/datasets/
        self.dataset_dir = os.environ.get(
            "MATMINER_DATA",
            os.path.abspath(os.path.join(self.current_dir, os.pardir))
        )

        # Shared set up for test_validate_dataset & test_fetch_external_dataset
        self.data_path = os.path.join(self.current_dir, "test_dataset.csv")
        self.dataset_metadata = RemoteFileMetadata(
            url="https://ndownloader.figshare.com/files/13039562",
            hash="c487f59ce0d48505c36633b4b202027d0c915474b081e8fb0bde8d5474ee59a1"
        )

    def test_validate_dataset(self):
        if os.path.exists(self.data_path):
            os.remove(self.data_path)

        with self.assertRaises(IOError):
            validate_dataset(self.data_path, self.dataset_metadata,
                             download_if_missing=False)

        with self.assertRaises(IOError):
            validate_dataset(self.data_path, dataset_metadata=None,
                             download_if_missing=False)

        with self.assertRaises(ValueError):
            validate_dataset(self.data_path, dataset_metadata=None,
                             download_if_missing=True)

        invalid_hash_metadata = RemoteFileMetadata(
            url=self.dataset_metadata.url,
            hash="!@#$%^&*()"
        )
        with self.assertRaises(IOError):
            validate_dataset(self.data_path, invalid_hash_metadata,
                             download_if_missing=True)
        os.remove(self.data_path)

        validate_dataset(self.data_path, self.dataset_metadata,
                         download_if_missing=True)
        self.assertTrue(os.path.exists(self.data_path))

    def test_fetch_external_dataset(self):
        if os.path.exists(self.data_path):
            os.remove(self.data_path)

        fetch_external_dataset(self.dataset_metadata.url, self.data_path)
        self.assertTrue(os.path.exists(self.data_path))

    def test_elastic_tensor(self):
        # Test that the dataset is downloadable, also get integrity check
        # from internal check against file hash
        data_path = os.path.join(self.dataset_dir, "elastic_tensor.csv")
        if os.path.exists(data_path):
            os.remove(data_path)

        load_elastic_tensor()
        self.assertTrue(os.path.exists(data_path))

        # Test that data is now available and properly formatted
        df = load_elastic_tensor(download_if_missing=False)
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
        df = load_elastic_tensor(include_metadata=True,
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

        load_piezoelectric_tensor()
        self.assertTrue(os.path.exists(data_path))

        # Test that data is now available and properly formatted
        df = load_piezoelectric_tensor(download_if_missing=False)
        self.assertEqual(len(df), 941)
        self.assertEqual(type(df['piezoelectric_tensor'][0]), np.ndarray)
        self.assertEqual(type(df['structure'][0]), Structure)
        column_headers = ['material_id', 'formula',
                          'nsites', 'point_group', 'space_group', 'volume',
                          'structure', 'eij_max', 'v_max', 'piezoelectric_tensor']
        self.assertEqual(list(df), column_headers)
        df = load_piezoelectric_tensor(include_metadata=True,
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

        load_dielectric_constant()
        self.assertTrue(os.path.exists(data_path))

        # Test that data is now available and properly formatted
        df = load_dielectric_constant(download_if_missing=False)
        self.assertEqual(type(df['structure'][0]), Structure)
        self.assertEqual(len(df), 1056)
        column_headers = ['material_id', 'formula',
                          'nsites', 'space_group', 'volume',
                          'structure',
                          'band_gap', 'e_electronic', 'e_total',
                          'n', 'poly_electronic',
                          'poly_total', 'pot_ferroelectric']
        self.assertEqual(list(df), column_headers)
        df = load_dielectric_constant(include_metadata=True,
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

        load_flla()
        self.assertTrue(os.path.exists(data_path))

        # Test that data is now available and properly formatted
        df = load_flla(download_if_missing=False)
        self.assertEqual(type(df['structure'][0]), Structure)
        self.assertEqual(len(df), 3938)
        column_headers = ['material_id', 'e_above_hull', 'formula',
                          'nsites', 'structure', 'formation_energy',
                          'formation_energy_per_atom']
        self.assertEqual(list(df), column_headers)

        os.remove(data_path)


if __name__ == "__main__":
    unittest.main()
