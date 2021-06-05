import os
import shutil
import filecmp
import unittest
import json

import pandas as pd

from monty.io import zopen

from pymatgen.core import Structure, Lattice
from pymatgen.util.testing import PymatgenTest

from matminer.utils.io import load_dataframe_from_json, store_dataframe_as_json

test_dir = os.path.dirname(__file__)


def generate_json_files():
    diamond = Structure(
        Lattice([[2.189, 0, 1.264], [0.73, 2.064, 1.264], [0, 0, 2.528]]),
        ["C0+", "C0+"],
        [[2.554, 1.806, 4.423], [0.365, 0.258, 0.632]],
        validate_proximity=False,
        to_unit_cell=False,
        coords_are_cartesian=True,
        site_properties=None,
    )
    df = pd.DataFrame(data={"structure": [diamond]})

    plain_file = os.path.join(test_dir, "dataframe.json")
    store_dataframe_as_json(df, plain_file)

    gz_file = os.path.join(test_dir, "dataframe.json.gz")
    store_dataframe_as_json(df, gz_file, compression="gz")

    bz2_file = os.path.join(test_dir, "dataframe.json.bz2")
    store_dataframe_as_json(df, bz2_file, compression="bz2")


class IOTest(PymatgenTest):
    def setUp(self):
        self.temp_folder = os.path.join(test_dir, "gzip_dir")
        os.mkdir(self.temp_folder)

        self.diamond = Structure(
            Lattice([[2.189, 0, 1.264], [0.73, 2.064, 1.264], [0, 0, 2.528]]),
            ["C0+", "C0+"],
            [[2.554, 1.806, 4.423], [0.365, 0.258, 0.632]],
            validate_proximity=False,
            to_unit_cell=False,
            coords_are_cartesian=True,
            site_properties=None,
        )
        self.df = pd.DataFrame(data={"structure": [self.diamond]})

    def test_store_dataframe_as_json(self):

        # check write produces correct file
        temp_file = os.path.join(self.temp_folder, "test_dataframe.json")
        test_file = os.path.join(test_dir, "dataframe.json")
        store_dataframe_as_json(self.df, temp_file)

        with zopen(temp_file, "rb") as f:
            temp_data = json.load(f)

        with zopen(test_file, "rb") as f:
            test_data = json.load(f)

        # remove version otherwise this will have to be updated everytime
        # the pymatgen version changes
        temp_data["data"][0][0].pop("@version")
        test_data["data"][0][0].pop("@version")

        self.assertDictsAlmostEqual(temp_data, test_data)

        # check writing gzipped json (comparing hashes doesn't work) so have to
        # compare contents
        temp_file = os.path.join(self.temp_folder, "test_dataframe.json.gz")
        test_file = os.path.join(test_dir, "dataframe.json.gz")
        store_dataframe_as_json(self.df, temp_file, compression="gz")

        with zopen(temp_file, "rb") as f:
            temp_data = json.load(f)

        with zopen(test_file, "rb") as f:
            test_data = json.load(f)

        temp_data["data"][0][0].pop("@version")
        test_data["data"][0][0].pop("@version")

        self.assertDictsAlmostEqual(temp_data, test_data)

        # check writing bz2 compressed json (comparing hashes doesn't work)
        # check writing gzipped json (comparing hashes doesn't work) so have to
        # compare contents
        temp_file = os.path.join(self.temp_folder, "test_dataframe.json.bz2")
        test_file = os.path.join(test_dir, "dataframe.json.bz2")
        store_dataframe_as_json(self.df, temp_file, compression="bz2")

        with zopen(temp_file, "rb") as f:
            temp_data = json.load(f)

        with zopen(test_file, "rb") as f:
            test_data = json.load(f)

        temp_data["data"][0][0].pop("@version")
        test_data["data"][0][0].pop("@version")

        self.assertDictsAlmostEqual(temp_data, test_data)

    def test_load_dataframe_from_json(self):

        df = load_dataframe_from_json(os.path.join(test_dir, "dataframe.json"))
        self.assertTrue(self.diamond == df["structure"][0], "Dataframe contents do not match")

        df = load_dataframe_from_json(os.path.join(test_dir, "dataframe.json.gz"))
        self.assertTrue(self.diamond == df["structure"][0], "Dataframe contents do not match")

        df = load_dataframe_from_json(os.path.join(test_dir, "dataframe.json.bz2"))
        self.assertTrue(self.diamond == df["structure"][0], "Dataframe contents do not match")

    def tearDown(self):
        shutil.rmtree(self.temp_folder)


if __name__ == "__main__":
    # generate_json_files()
    unittest.main()
