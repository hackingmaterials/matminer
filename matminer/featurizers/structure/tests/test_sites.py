import unittest

import numpy as np

from matminer.featurizers.site import SiteElementalProperty
from matminer.featurizers.structure.sites import (
    PartialsSiteStatsFingerprint,
    SiteStatsFingerprint,
)
from matminer.featurizers.structure.tests.base import StructureFeaturesTest


class StructureSitesFeaturesTest(StructureFeaturesTest):
    def test_sitestatsfingerprint(self):
        # Test matrix.
        op_struct_fp = SiteStatsFingerprint.from_preset("OPSiteFingerprint", stats=None)
        opvals = op_struct_fp.featurize(self.diamond)
        _ = op_struct_fp.feature_labels()
        self.assertAlmostEqual(opvals[10][0], 0.9995, places=7)
        self.assertAlmostEqual(opvals[10][1], 0.9995, places=7)
        opvals = op_struct_fp.featurize(self.nacl)
        self.assertAlmostEqual(opvals[18][0], 0.9995, places=7)
        self.assertAlmostEqual(opvals[18][1], 0.9995, places=7)
        opvals = op_struct_fp.featurize(self.cscl)
        self.assertAlmostEqual(opvals[22][0], 0.9995, places=7)
        self.assertAlmostEqual(opvals[22][1], 0.9995, places=7)

        # Test stats.
        op_struct_fp = SiteStatsFingerprint.from_preset("OPSiteFingerprint")
        opvals = op_struct_fp.featurize(self.diamond)
        self.assertAlmostEqual(opvals[0], 0.0005, places=7)
        self.assertAlmostEqual(opvals[1], 0, places=7)
        self.assertAlmostEqual(opvals[2], 0.0005, places=7)
        self.assertAlmostEqual(opvals[3], 0.0, places=7)
        self.assertAlmostEqual(opvals[4], 0.0005, places=7)
        self.assertAlmostEqual(opvals[18], 0.0805, places=7)
        self.assertAlmostEqual(opvals[20], 0.9995, places=7)
        self.assertAlmostEqual(opvals[21], 0, places=7)
        self.assertAlmostEqual(opvals[22], 0.0075, places=7)
        self.assertAlmostEqual(opvals[24], 0.2355, places=7)
        self.assertAlmostEqual(opvals[-1], 0.0, places=7)

        # Test coordination number
        cn_fp = SiteStatsFingerprint.from_preset("JmolNN", stats=("mean",))
        cn_vals = cn_fp.featurize(self.diamond)
        self.assertEqual(cn_vals[0], 4.0)

        # Test the covariance
        prop_fp = SiteStatsFingerprint(
            SiteElementalProperty(properties=["Number", "AtomicWeight"]),
            stats=["mean"],
            covariance=True,
        )

        # Test the feature labels
        labels = prop_fp.feature_labels()
        self.assertEqual(3, len(labels))

        #  Test a structure with all the same type (cov should be zero)
        features = prop_fp.featurize(self.diamond)
        np.testing.assert_array_almost_equal(features, [6, 12.0107, 0])

        #  Test a structure with only one atom (cov should be zero too)
        features = prop_fp.featurize(self.sc)
        np.testing.assert_array_almost_equal([13, 26.9815386, 0], features)

        #  Test a structure with nonzero covariance
        features = prop_fp.featurize(self.nacl)
        np.testing.assert_array_almost_equal([14, 29.22138464, 37.38969216], features)

        # Test soap site featurizer
        soap_fp = SiteStatsFingerprint.from_preset("SOAP_formation_energy")
        soap_fp.fit([self.sc, self.diamond, self.nacl])
        feats = soap_fp.featurize(self.diamond)
        self.assertEqual(len(feats), 9504)
        self.assertAlmostEqual(feats[0], 0.4412608, places=5)
        self.assertAlmostEqual(feats[1], 0.0)
        self.assertAlmostEqual(np.sum(feats), 207.88194724, places=5)

    def test_ward_prb_2017_lpd(self):
        """Test the local property difference attributes from Ward 2017"""
        f = SiteStatsFingerprint.from_preset("LocalPropertyDifference_ward-prb-2017")

        # Test diamond
        features = f.featurize(self.diamond)
        np.testing.assert_array_almost_equal(features, [0] * (22 * 5))
        features = f.featurize(self.diamond_no_oxi)
        np.testing.assert_array_almost_equal(features, [0] * (22 * 5))

        # Test CsCl
        big_face_area = np.sqrt(3) * 3 / 2 * (2 / 4 / 4)
        small_face_area = 0.125
        big_face_diff = 55 - 17
        features = f.featurize(self.cscl)
        labels = f.feature_labels()
        my_label = "mean local difference in Number"
        self.assertAlmostEqual(
            (8 * big_face_area * big_face_diff) / (8 * big_face_area + 6 * small_face_area),
            features[labels.index(my_label)],
            places=3,
        )
        my_label = "range local difference in Electronegativity"
        self.assertAlmostEqual(0, features[labels.index(my_label)], places=3)

    def test_ward_prb_2017_efftcn(self):
        """Test the effective coordination number attributes of Ward 2017"""
        f = SiteStatsFingerprint.from_preset("CoordinationNumber_ward-prb-2017")

        # Test Ni3Al
        features = f.featurize(self.ni3al)
        labels = f.feature_labels()
        my_label = "mean CN_VoronoiNN"
        self.assertAlmostEqual(12, features[labels.index(my_label)])
        np.testing.assert_array_almost_equal([12, 12, 0, 12, 0], features)


class PartialStructureSitesFeaturesTest(StructureFeaturesTest):
    def test_partialsitestatsfingerprint(self):
        # Test matrix.
        op_struct_fp = PartialsSiteStatsFingerprint.from_preset("OPSiteFingerprint", stats=None)

        op_struct_fp.fit([self.diamond])
        opvals = op_struct_fp.featurize(self.diamond)
        _ = op_struct_fp.feature_labels()
        self.assertAlmostEqual(opvals[10][0], 0.9995, places=7)
        self.assertAlmostEqual(opvals[10][1], 0.9995, places=7)

        op_struct_fp.fit([self.nacl])
        opvals = op_struct_fp.featurize(self.nacl)
        self.assertAlmostEqual(opvals[18][0], 0.9995, places=7)
        self.assertAlmostEqual(opvals[18][1], 0.9995, places=7)

        op_struct_fp.fit([self.cscl])
        opvals = op_struct_fp.featurize(self.cscl)
        self.assertAlmostEqual(opvals[22][0], 0.9995, places=7)
        self.assertAlmostEqual(opvals[22][1], 0.9995, places=7)

        # Test stats.
        op_struct_fp = PartialsSiteStatsFingerprint.from_preset("OPSiteFingerprint")
        op_struct_fp.fit([self.diamond])
        opvals = op_struct_fp.featurize(self.diamond)
        self.assertAlmostEqual(opvals[0], 0.0005, places=7)
        self.assertAlmostEqual(opvals[1], 0, places=7)
        self.assertAlmostEqual(opvals[2], 0.0005, places=7)
        self.assertAlmostEqual(opvals[3], 0.0, places=7)
        self.assertAlmostEqual(opvals[4], 0.0005, places=7)
        self.assertAlmostEqual(opvals[18], 0.0805, places=7)
        self.assertAlmostEqual(opvals[20], 0.9995, places=7)
        self.assertAlmostEqual(opvals[21], 0, places=7)
        self.assertAlmostEqual(opvals[22], 0.0075, places=7)
        self.assertAlmostEqual(opvals[24], 0.2355, places=7)
        self.assertAlmostEqual(opvals[-1], 0.0, places=7)

        # Test coordination number
        cn_fp = PartialsSiteStatsFingerprint.from_preset("JmolNN", stats=("mean",))
        cn_fp.fit([self.diamond])
        cn_vals = cn_fp.featurize(self.diamond)
        self.assertEqual(cn_vals[0], 4.0)

        # Test the covariance
        prop_fp = PartialsSiteStatsFingerprint(
            SiteElementalProperty(properties=["Number", "AtomicWeight"]),
            stats=["mean"],
            covariance=True,
        )

        # Test the feature labels
        prop_fp.fit([self.diamond])
        labels = prop_fp.feature_labels()
        self.assertEqual(3, len(labels))

        #  Test a structure with all the same type (cov should be zero)
        prop_fp.fit([self.diamond])
        features = prop_fp.featurize(self.diamond)
        np.testing.assert_array_almost_equal(features, [6, 12.0107, 0])

        #  Test a structure with only one atom (cov should be zero too)
        prop_fp.fit([self.sc])
        features = prop_fp.featurize(self.sc)
        np.testing.assert_array_almost_equal([13, 26.9815386, 0], features)

        #  Test a structure with nonzero covariance
        prop_fp.fit([self.nacl])
        features = prop_fp.featurize(self.nacl)
        np.testing.assert_array_almost_equal([11, 22.9897693, np.nan, 17, 35.453, np.nan], features)

    def test_ward_prb_2017_lpd(self):
        """Test the local property difference attributes from Ward 2017"""
        f = PartialsSiteStatsFingerprint.from_preset("LocalPropertyDifference_ward-prb-2017")

        # Test diamond
        f.fit([self.diamond])
        features = f.featurize(self.diamond)
        np.testing.assert_array_almost_equal(features, [0] * (22 * 5))
        features = f.featurize(self.diamond_no_oxi)
        np.testing.assert_array_almost_equal(features, [0] * (22 * 5))

        # Test CsCl
        f.fit([self.cscl])
        big_face_area = np.sqrt(3) * 3 / 2 * (2 / 4 / 4)
        small_face_area = 0.125
        big_face_diff = 55 - 17
        features = f.featurize(self.cscl)
        labels = f.feature_labels()
        my_label = "Cs mean local difference in Number"
        self.assertAlmostEqual(
            (8 * big_face_area * big_face_diff) / (8 * big_face_area + 6 * small_face_area),
            features[labels.index(my_label)],
            places=3,
        )
        my_label = "Cs range local difference in Electronegativity"
        self.assertAlmostEqual(0, features[labels.index(my_label)], places=3)

    def test_ward_prb_2017_efftcn(self):
        """Test the effective coordination number attributes of Ward 2017"""
        f = PartialsSiteStatsFingerprint.from_preset("CoordinationNumber_ward-prb-2017")

        # Test Ni3Al
        f.fit([self.ni3al])
        features = f.featurize(self.ni3al)
        labels = f.feature_labels()
        self.assertAlmostEqual(12, features[labels.index("Al mean CN_VoronoiNN")])
        self.assertAlmostEqual(12, features[labels.index("Ni mean CN_VoronoiNN")])
        np.testing.assert_array_almost_equal([12, 12, 0, 12, 0] * 2, features)


if __name__ == "__main__":
    unittest.main()
