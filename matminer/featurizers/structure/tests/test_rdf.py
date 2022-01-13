import unittest

import numpy as np
import pandas as pd

from matminer.featurizers.structure.rdf import (
    ElectronicRadialDistributionFunction,
    PartialRadialDistributionFunction,
    RadialDistributionFunction,
    get_rdf_bin_labels,
)
from matminer.featurizers.structure.tests.base import StructureFeaturesTest


class StructureRDFTest(StructureFeaturesTest):
    def test_rdf_and_peaks(self):
        # Test diamond
        rdf = RadialDistributionFunction()
        diamond_rdf = rdf.featurize(self.diamond)

        # Prechecking test
        self.assertTrue(rdf.precheck(self.diamond))
        self.assertFalse(rdf.precheck(self.disordered_diamond))

        # Make sure it the last bin is cutoff-bin_max
        self.assertAlmostEqual(max(rdf.bin_distances), 19.9)

        # Verify bin sizes
        self.assertEqual(len(diamond_rdf), 200)

        # Make sure it gets all of the peaks
        self.assertEqual(np.count_nonzero(diamond_rdf), 116)

        # Check the values for a few individual peaks
        self.assertAlmostEqual(diamond_rdf[int(round(1.5 / 0.1))], 15.12755155)
        self.assertAlmostEqual(diamond_rdf[int(round(2.9 / 0.1))], 12.53193948)
        self.assertAlmostEqual(diamond_rdf[int(round(19.9 / 0.1))], 0.822126129)

        # Check the feature labels make sense
        self.assertEqual(rdf.feature_labels()[0], "rdf [0.00000 - 0.10000]A")
        self.assertEqual(rdf.feature_labels()[9], "rdf [0.90000 - 1.00000]A")

        # Repeat test with NaCl (omitting comments). Altering cutoff distance
        rdf2 = RadialDistributionFunction(cutoff=10)
        nacl_rdf = rdf2.featurize(self.nacl)
        self.assertAlmostEqual(max(rdf2.bin_distances), 9.9)
        self.assertEqual(len(nacl_rdf), 100)
        self.assertEqual(np.count_nonzero(nacl_rdf), 11)
        self.assertAlmostEqual(nacl_rdf[int(round(2.8 / 0.1))], 27.09214168)
        self.assertAlmostEqual(nacl_rdf[int(round(4.0 / 0.1))], 26.83338723)
        self.assertAlmostEqual(nacl_rdf[int(round(9.8 / 0.1))], 3.024406467)

        # Repeat test with CsCl. Altering cutoff distance and bin_size
        rdf3 = RadialDistributionFunction(cutoff=8, bin_size=0.5)
        cscl_rdf = rdf3.featurize(self.cscl)
        self.assertAlmostEqual(max(rdf3.bin_distances), 7.5)
        self.assertEqual(len(cscl_rdf), 16)
        self.assertEqual(np.count_nonzero(cscl_rdf), 5)
        self.assertAlmostEqual(cscl_rdf[int(round(3.5 / 0.5))], 6.741265585)
        self.assertAlmostEqual(cscl_rdf[int(round(4.0 / 0.5))], 3.937582548)
        self.assertAlmostEqual(cscl_rdf[int(round(7.0 / 0.5))], 1.805505363)

    def test_prdf(self):
        # Test a few peaks in diamond
        # These expected numbers were derived by performing
        # the calculation in another code
        distances, prdf = PartialRadialDistributionFunction().compute_prdf(self.diamond)

        # Check prechecking
        prdf_obj = PartialRadialDistributionFunction()
        self.assertTrue(prdf_obj.precheck(self.diamond))
        self.assertFalse(prdf_obj.precheck(self.disordered_diamond))

        self.assertEqual(len(prdf.values()), 1)
        self.assertAlmostEqual(prdf[("C", "C")][int(round(1.4 / 0.1))], 0)
        self.assertAlmostEqual(prdf[("C", "C")][int(round(1.5 / 0.1))], 1.32445167622)
        self.assertAlmostEqual(max(distances), 19.9)
        self.assertAlmostEqual(prdf[("C", "C")][int(round(19.9 / 0.1))], 0.07197902)

        # Test a few peaks in CsCl, make sure it gets all types correctly
        distances, prdf = PartialRadialDistributionFunction(cutoff=10).compute_prdf(self.cscl)
        self.assertEqual(len(prdf.values()), 4)
        self.assertAlmostEqual(max(distances), 9.9)
        self.assertAlmostEqual(prdf[("Cs", "Cl")][int(round(3.6 / 0.1))], 0.477823197)
        self.assertAlmostEqual(prdf[("Cl", "Cs")][int(round(3.6 / 0.1))], 0.477823197)
        self.assertAlmostEqual(prdf[("Cs", "Cs")][int(round(3.6 / 0.1))], 0)

        # Do Ni3Al, make sure it captures the antisymmetry of Ni/Al sites
        distances, prdf = PartialRadialDistributionFunction(cutoff=10, bin_size=0.5).compute_prdf(self.ni3al)
        self.assertEqual(len(prdf.values()), 4)
        self.assertAlmostEqual(prdf[("Ni", "Al")][int(round(2 / 0.5))], 0.125236677)
        self.assertAlmostEqual(prdf[("Al", "Ni")][int(round(2 / 0.5))], 0.37571003)
        self.assertAlmostEqual(prdf[("Al", "Al")][int(round(2 / 0.5))], 0)

        # Check the fit operation
        featurizer = PartialRadialDistributionFunction()
        featurizer.fit([self.diamond, self.cscl, self.ni3al])
        self.assertEqual({"Cs", "Cl", "C", "Ni", "Al"}, set(featurizer.elements_))

        featurizer.exclude_elems = ["Cs", "Al"]
        featurizer.fit([self.diamond, self.cscl, self.ni3al])
        self.assertEqual({"Cl", "C", "Ni"}, set(featurizer.elements_))

        featurizer.include_elems = ["H"]
        featurizer.fit([self.diamond, self.cscl, self.ni3al])
        self.assertEqual({"H", "Cl", "C", "Ni"}, set(featurizer.elements_))

        # Check the feature labels
        featurizer.exclude_elems = ()
        featurizer.include_elems = ()
        featurizer.elements_ = ["Al", "Ni"]
        labels = featurizer.feature_labels()
        n_bins = len(featurizer._make_bins()) - 1

        self.assertEqual(3 * n_bins, len(labels))
        self.assertIn("Al-Ni PRDF r=0.00-0.10", labels)

        # Check the featurize method
        featurizer.elements_ = ["C"]
        features = featurizer.featurize(self.diamond)
        prdf = featurizer.compute_prdf(self.diamond)[1]
        self.assertArrayAlmostEqual(features, prdf[("C", "C")])

        # Check the featurize_dataframe
        df = pd.DataFrame.from_dict({"structure": [self.diamond, self.cscl]})
        featurizer.fit(df["structure"])
        df = featurizer.featurize_dataframe(df, col_id="structure")
        self.assertEqual(df["Cs-Cl PRDF r=0.00-0.10"][0], 0.0)
        self.assertAlmostEqual(df["Cl-Cl PRDF r=19.70-19.80"][1], 0.049, 3)
        self.assertEqual(df["Cl-Cl PRDF r=19.90-20.00"][0], 0.0)

        # Make sure labels and features are in the same order
        featurizer.elements_ = ["Al", "Ni"]
        features = featurizer.featurize(self.ni3al)
        labels = featurizer.feature_labels()
        prdf = featurizer.compute_prdf(self.ni3al)[1]
        self.assertEqual((n_bins * 3,), features.shape)
        self.assertTrue(labels[0].startswith("Al-Al"))
        self.assertTrue(labels[n_bins].startswith("Al-Ni"))
        self.assertTrue(labels[2 * n_bins].startswith("Ni-Ni"))
        self.assertArrayAlmostEqual(
            features,
            np.hstack([prdf[("Al", "Al")], prdf[("Al", "Ni")], prdf[("Ni", "Ni")]]),
        )

    def test_redf(self):
        # Test prechecking
        erdf = ElectronicRadialDistributionFunction(cutoff=10, dr=0.05)
        self.assertTrue(erdf.precheck(self.diamond))
        self.assertFalse(erdf.precheck(self.disordered_diamond))
        self.assertFalse(erdf.precheck(self.diamond_no_oxi))

        # C has oxi state of 0 in diamond, so we expect them all to be 0
        d = erdf.featurize(self.diamond)
        self.assertAlmostEqual(erdf.distances[0], 0)
        self.assertAlmostEqual(erdf.distances[1], 0.05)
        self.assertFalse(np.asarray(d).any())

        d = erdf.featurize(self.nacl)
        self.assertAlmostEqual(erdf.distances[0], 0)
        self.assertAlmostEqual(erdf.distances[1], 0.05)
        self.assertTrue(np.asarray(d).any())
        self.assertAlmostEqual(d[-4], 0.81151636)
        self.assertAlmostEqual(d[-13], -2.54280359)
        self.assertAlmostEqual(d[56], -2.10838136)

        d = erdf.featurize(self.cscl)
        self.assertAlmostEqual(erdf.distances[0], 0)
        self.assertAlmostEqual(erdf.distances[1], 0.05)
        self.assertAlmostEqual(d[72], -2.19472661)
        self.assertAlmostEqual(d[-13], 2.55004188)

    def test_get_rdf_bin_labels(self):
        bin_distances = [1, 2, 3, 4, 5]
        cutoff = 6
        flabels = get_rdf_bin_labels(bin_distances, cutoff)
        self.assertEqual(flabels[0], "[1.00000 - 2.00000]")
        self.assertEqual(flabels[2], "[3.00000 - 4.00000]")
        self.assertEqual(flabels[-1], "[5.00000 - 6.00000]")


if __name__ == "__main__":
    unittest.main()
