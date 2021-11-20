"""
Test Element Mover's 2D Distance Matrix via network simplex and "wasserstein" methods.

This test ensures that the fast implementation "wasserstein" produces "close" values
to the original network simplex method.
"""
import unittest

from os.path import join, dirname, relpath

from numpy import genfromtxt
from numpy.testing import assert_allclose
import pandas as pd

from matminer.featurizers.ElM2D_ import ElM2DFeaturizer

target = "cuda"


class Testing(unittest.TestCase):
    def test_dm_close(self):
        mapper = ElM2DFeaturizer()
        # df = pd.read_csv("train-debug.csv")
        df = pd.read_csv(join(dirname(relpath(__file__)), "stable-mp-500.csv"))
        formulas = df["formula"]
        nformulas = 500
        sub_formulas = formulas[:nformulas]

        mapper.fit(sub_formulas, target=target)
        dm_wasserstein = mapper.dm

        dm_check = genfromtxt(
            join(dirname(relpath(__file__)), "tests", "ElM2D_check.csv")
        )

        # 500 x 500 distance matrix
        if nformulas > 500:
            raise ValueError(
                "nformulas>500, should be <=500 (received: {})".format(nformulas)
            )

        assert_allclose(
            dm_wasserstein,
            dm_check,
            atol=1e-3,
            err_msg="wasserstein did not match ElM2D (0.4.0) with ElMD (0-4-3).",
        )


if __name__ == "__main__":
    unittest.main()
