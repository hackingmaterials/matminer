import unittest
import os
from itertools import product

from matminer.datasets.tests import DataSetTest
from matminer.datasets.dataset_retrieval import load_dataset, get_available_datasets


class DataRetrievalTest(DataSetTest):
    # This test case only checks the dataset loaders exceptions and a simple
    # case, for more extensive tests of individual datasets see the
    # test_load_"dataset name" functions in test_datasets.py
    def test_load_dataset(self):
        # Can't find dataset or similar
        with self.assertRaises(ValueError):
            load_dataset("not_real_dataset")
        # Finds similar
        with self.assertRaises(ValueError):
            load_dataset("tensor")
        # Actual dataset is subset of passed dataset name
        dataset_name = sorted(self.dataset_dict.keys())[0]
        with self.assertRaises(ValueError):
            load_dataset("a" + dataset_name + "a")

        dataset_filename = (dataset_name + "."
                            + self.dataset_dict[dataset_name]["file_type"])
        data_home = os.path.expanduser("~")
        dataset_path = os.path.join(data_home, dataset_filename)
        if os.path.exists(dataset_path):
            os.remove(dataset_path)

        load_dataset(dataset_name, data_home)
        self.assertTrue(os.path.exists(data_home))

    def test_available_datasets(self):
        # Go over all parameter combinations,
        # for each check that returned dataset is correct
        for parameter_combo in product([True, False], [True, False],
                                       ['alphabetical', 'num_entries']):
            datasets = get_available_datasets(*parameter_combo)
            if parameter_combo[2] == 'alphabetical':
                self.assertEqual(datasets, sorted(self.dataset_names))
            else:
                self.assertEqual(
                    datasets,
                    sorted(self.dataset_names,
                           key=lambda x: self.dataset_dict[x]['num_entries'],
                           reverse=True)
                )


if __name__ == "__main__":
    unittest.main()
