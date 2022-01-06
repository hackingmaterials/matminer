import os
import unittest

import pandas as pd

from matminer.datasets.tests.base import DatasetTest
from matminer.datasets.utils import (
    _fetch_external_dataset,
    _get_data_home,
    _get_file_sha256_hash,
    _load_dataset_dict,
    _read_dataframe_from_file,
    _validate_dataset,
)


class UtilsTest(DatasetTest):
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
            for item in ["file_type", "url", "hash", "reference", "description"]:
                self.assertIsInstance(value[item], str)
            # Make sure int attributes have int values
            self.assertIsInstance(value["num_entries"], int)
            # Make sure refs are in a list and are strings
            self.assertIsInstance(value["bibtex_refs"], list)
            for ref in value["bibtex_refs"]:
                self.assertIsInstance(ref, str)
            # Make sure columns is a dict and it has string valued entries
            self.assertIsInstance(value["columns"], dict)
            for column_name, column_description in value["columns"].items():
                self.assertIsInstance(column_name, str)
                self.assertIsInstance(column_description, str)

    def test_get_data_home(self):
        home = _get_data_home()
        self.assertEqual(home, self.dataset_dir)
        specified_home = _get_data_home("/some/specified/path")
        self.assertEqual(specified_home, "/some/specified/path")

    def test_validate_dataset(self):
        if os.path.exists(self._path):
            os.remove(self._path)

        with self.assertRaises(IOError):
            _validate_dataset(self._path, self._url, self._hash, download_if_missing=False)

        # Check to make sure the IOError takes precedence over the ValueError
        with self.assertRaises(IOError):
            _validate_dataset(self._path, url=None, file_hash=self._hash, download_if_missing=False)

        with self.assertRaises(ValueError):
            _validate_dataset(self._path, url=None, file_hash=self._hash, download_if_missing=True)

        with self.assertRaises(UserWarning):
            _validate_dataset(self._path, self._url, file_hash="!@#$%^&*", download_if_missing=True)
        os.remove(self._path)

        _validate_dataset(self._path, self._url, self._hash, download_if_missing=True)
        self.assertTrue(os.path.exists(self._path))
        os.remove(self._path)

        _validate_dataset(self._path, self._url, file_hash=None, download_if_missing=True)
        self.assertTrue(os.path.exists(self._path))
        os.remove(self._path)

    def test_fetch_external_dataset(self):
        if os.path.exists(self._path):
            os.remove(self._path)

        _fetch_external_dataset(self._url, self._path)
        self.assertTrue(os.path.exists(self._path))
        os.remove(self._path)

    def test_get_file_sha256_hash(self):
        if not os.path.exists(self._path):
            _fetch_external_dataset(self._url, self._path)

        self.assertTrue(_get_file_sha256_hash(self._path) == self._hash)
        os.remove(self._path)

    def test_read_dataframe_from_file(self):
        if not os.path.exists(self._path):
            _fetch_external_dataset(self._url, self._path)

        self.assertTrue(isinstance(_read_dataframe_from_file(self._path), pd.DataFrame))

        with self.assertRaises(ValueError):
            _read_dataframe_from_file("nonexistent.txt")


if __name__ == "__main__":
    unittest.main()
