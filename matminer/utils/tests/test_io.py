import os
import shutil
import filecmp
import unittest
import json

import pandas as pd

from monty.io import zopen

from pymatgen import Structure, Lattice
from pymatgen.util.testing import PymatgenTest

from matminer.utils.io import load_dataframe_from_json, store_dataframe_as_json

test_dir = os.path.dirname(__file__)


class IOTest(PymatgenTest):

    def setUp(self):
        self.temp_folder = os.path.join(test_dir, "gzip_dir")
        os.mkdir(self.temp_folder)

        self.diamond = Structure(
            Lattice([[2.189, 0, 1.264], [0.73, 2.064, 1.264], [0, 0, 2.528]]),
            ["C0+", "C0+"], [[2.554, 1.806, 4.423], [0.365, 0.258, 0.632]],
            validate_proximity=False,
            to_unit_cell=False, coords_are_cartesian=True,
            site_properties=None
        )
        self.df = pd.DataFrame(data={'structure': [self.diamond]})

    def test_store_dataframe_as_json(self):

        # check write produces correct file
        temp_file = os.path.join(self.temp_folder, 'test_dataframe.json')
        test_file = os.path.join(test_dir, "dataframe.json")
        store_dataframe_as_json(self.df, temp_file)

        self.assertTrue(
            filecmp.cmp(temp_file, test_file), "Json files do not match.")

        # check writing gzipped json (comparing hashes doesn't work) so have to
        # compare contents
        temp_file = os.path.join(self.temp_folder, 'test_dataframe.json.gz')
        test_file = os.path.join(test_dir, "dataframe.json.gz")
        store_dataframe_as_json(self.df, temp_file, compression='gz')

        with zopen(temp_file, 'rb') as f:
            temp_data = json.load(f)

        with zopen(test_file, 'rb') as f:
            test_data = json.load(f)

        self.assertTrue(temp_data == test_data,
                        "Compressed json files do not match.")

        # check writing bz2 compressed json (comparing hashes doesn't work)
        # check writing gzipped json (comparing hashes doesn't work) so have to
        # compare contents
        temp_file = os.path.join(self.temp_folder, 'test_dataframe.json.bz2')
        test_file = os.path.join(test_dir, "dataframe.json.bz2")
        store_dataframe_as_json(self.df, temp_file, compression='bz2')

        with zopen(temp_file, 'rb') as f:
            temp_data = json.load(f)

        with zopen(test_file, 'rb') as f:
            test_data = json.load(f)

        self.assertTrue(temp_data == test_data,
                        "Compressed json files do not match.")

    def test_load_dataframe_from_json(self):

        df = load_dataframe_from_json(os.path.join(test_dir, 'dataframe.json'))
        self.assertTrue(self.diamond == df['structure'][0],
                        "Dataframe contents do not match")

        df = load_dataframe_from_json(os.path.join(test_dir, 'dataframe.json.gz'))
        self.assertTrue(self.diamond == df['structure'][0],
                        "Dataframe contents do not match")

        df = load_dataframe_from_json(os.path.join(test_dir, 'dataframe.json.bz2'))
        self.assertTrue(self.diamond == df['structure'][0],
                        "Dataframe contents do not match")

    def tearDown(self):
        shutil.rmtree(self.temp_folder)


if __name__ == '__main__':
    unittest.main()
