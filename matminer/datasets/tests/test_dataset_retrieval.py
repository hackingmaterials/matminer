import unittest

from matminer.datasets.tests.base import DatasetTest
from matminer.datasets.dataset_retrieval import (
    load_dataset,
    get_available_datasets,
    get_dataset_attribute,
    get_dataset_description,
    get_dataset_num_entries,
    get_dataset_columns,
    get_dataset_reference,
    get_dataset_column_description,
    get_dataset_citations,
    get_all_dataset_info,
)


class DataRetrievalTest(DatasetTest):
    # This test case only checks the dataset loaders exceptions
    # For more extensive tests of individual datasets see the
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

    def test_print_available_datasets(self):
        # Go over all parameter combinations,
        # for each check that returned dataset is correct
        for sort_method in ["alphabetical", "num_entries"]:
            datasets = get_available_datasets(sort_method=sort_method)
            if sort_method == "alphabetical":
                self.assertEqual(datasets, sorted(self.dataset_names))
            else:
                self.assertEqual(
                    datasets,
                    sorted(
                        self.dataset_names,
                        key=lambda x: self.dataset_dict[x]["num_entries"],
                        reverse=True,
                    ),
                )

    def test_get_dataset_attribute(self):
        dataset_name = sorted(self.dataset_dict.keys())[0]
        file_type = get_dataset_attribute(dataset_name, "file_type")
        self.assertTrue(isinstance(file_type, str))

    def test_get_dataset_description(self):
        dataset_name = sorted(self.dataset_dict.keys())[0]
        attrib = get_dataset_description(dataset_name)
        self.assertTrue(isinstance(attrib, str))

    def test_get_dataset_num_entries(self):
        dataset_name = sorted(self.dataset_dict.keys())[0]
        attrib = get_dataset_num_entries(dataset_name)
        self.assertTrue(isinstance(attrib, int))

    def test_get_dataset_columns(self):
        dataset_name = sorted(self.dataset_dict.keys())[0]
        attrib = get_dataset_columns(dataset_name)
        self.assertTrue(isinstance(attrib, list))

    def test_get_dataset_reference(self):
        dataset_name = sorted(self.dataset_dict.keys())[0]
        attrib = get_dataset_reference(dataset_name)
        self.assertTrue(isinstance(attrib, str))

    def test_get_dataset_column_descriptions(self):
        dataset_name = sorted(self.dataset_dict.keys())[0]
        dataset_column = sorted(self.dataset_dict[dataset_name]["columns"].keys())[0]
        attrib = get_dataset_column_description(dataset_name, dataset_column)
        self.assertTrue(isinstance(attrib, str))

    def test_get_dataset_citations(self):
        dataset_name = sorted(self.dataset_dict.keys())[0]
        attrib = get_dataset_citations(dataset_name)
        self.assertTrue(isinstance(attrib, list))

    def test_get_all_dataset_info(self):
        dataset_name = sorted(self.dataset_dict.keys())[0]
        attrib = get_all_dataset_info(dataset_name)
        self.assertTrue(isinstance(attrib, str))


if __name__ == "__main__":
    unittest.main()
