import unittest
import os


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
        self._hash = "c487f59ce0d48505c36633b4b202027" \
                     "d0c915474b081e8fb0bde8d5474ee59a1"
