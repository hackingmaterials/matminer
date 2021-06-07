import unittest

import numpy as np
import pandas as pd
from pymatgen.util.testing import PymatgenTest

from matminer.featurizers.site.fingerprint import (
    AGNIFingerprints,
    OPSiteFingerprint,
    CrystalNNFingerprint,
    VoronoiFingerprint,
    ChemEnvSiteFingerprint,
)
from matminer.featurizers.site.tests.base import SiteFeaturizerTest


class FingerprintTests(SiteFeaturizerTest):
    def test_simple_cubic(self):
        """Test with an easy structure"""

        # Make sure direction-dependent fingerprints are zero
        agni = AGNIFingerprints(directions=["x", "y", "z"])

        features = agni.featurize(self.sc, 0)
        self.assertEqual(8 * 3, len(features))
        self.assertEqual(8 * 3, len(set(agni.feature_labels())))
        self.assertArrayAlmostEqual(
            [
                0,
            ]
            * 24,
            features,
        )

        # Compute the "atomic fingerprints"
        agni.directions = [None]
        agni.cutoff = 3.75  # To only get 6 neighbors to deal with

        features = agni.featurize(self.sc, 0)
        self.assertEqual(8, len(features))
        self.assertEqual(8, len(set(agni.feature_labels())))

        self.assertEqual(0.8, agni.etas[0])
        self.assertAlmostEqual(
            6 * np.exp(-((3.52 / 0.8) ** 2)) * 0.5 * (np.cos(np.pi * 3.52 / 3.75) + 1),
            features[0],
        )
        self.assertAlmostEqual(
            6 * np.exp(-((3.52 / 16) ** 2)) * 0.5 * (np.cos(np.pi * 3.52 / 3.75) + 1),
            features[-1],
        )

        # Test that passing etas to constructor works
        new_etas = np.logspace(-4, 2, 6)
        agni = AGNIFingerprints(directions=["x", "y", "z"], etas=new_etas)
        self.assertArrayAlmostEqual(new_etas, agni.etas)

    def test_off_center_cscl(self):
        agni = AGNIFingerprints(directions=[None, "x", "y", "z"], cutoff=4)

        # Compute the features on both sites
        site1 = agni.featurize(self.cscl, 0)
        site2 = agni.featurize(self.cscl, 1)

        # The atomic attributes should be equal
        self.assertArrayAlmostEqual(site1[:8], site2[:8])

        # The direction-dependent ones should be equal and opposite in sign
        self.assertArrayAlmostEqual(-1 * site1[8:], site2[8:])

        # Make sure the site-ones are as expected.
        right_dist = 4.209 * np.sqrt(0.45 ** 2 + 2 * 0.5 ** 2)
        right_xdist = 4.209 * 0.45
        left_dist = 4.209 * np.sqrt(0.55 ** 2 + 2 * 0.5 ** 2)
        left_xdist = 4.209 * 0.55
        self.assertAlmostEqual(
            4
            * (
                right_xdist
                / right_dist
                * np.exp(-((right_dist / 0.8) ** 2))
                * 0.5
                * (np.cos(np.pi * right_dist / 4) + 1)
                - left_xdist / left_dist * np.exp(-((left_dist / 0.8) ** 2)) * 0.5 * (np.cos(np.pi * left_dist / 4) + 1)
            ),
            site1[8],
        )

    def test_dataframe(self):
        data = pd.DataFrame({"strc": [self.cscl, self.cscl, self.sc], "site": [0, 1, 0]})

        agni = AGNIFingerprints()
        agni.featurize_dataframe(data, ["strc", "site"])

    def test_op_site_fingerprint(self):
        opsf = OPSiteFingerprint()
        l = opsf.feature_labels()
        t = [
            "sgl_bd CN_1",
            "L-shaped CN_2",
            "water-like CN_2",
            "bent 120 degrees CN_2",
            "bent 150 degrees CN_2",
            "linear CN_2",
            "trigonal planar CN_3",
            "trigonal non-coplanar CN_3",
            "T-shaped CN_3",
            "square co-planar CN_4",
            "tetrahedral CN_4",
            "rectangular see-saw-like CN_4",
            "see-saw-like CN_4",
            "trigonal pyramidal CN_4",
            "pentagonal planar CN_5",
            "square pyramidal CN_5",
            "trigonal bipyramidal CN_5",
            "hexagonal planar CN_6",
            "octahedral CN_6",
            "pentagonal pyramidal CN_6",
            "hexagonal pyramidal CN_7",
            "pentagonal bipyramidal CN_7",
            "body-centered cubic CN_8",
            "hexagonal bipyramidal CN_8",
            "q2 CN_9",
            "q4 CN_9",
            "q6 CN_9",
            "q2 CN_10",
            "q4 CN_10",
            "q6 CN_10",
            "q2 CN_11",
            "q4 CN_11",
            "q6 CN_11",
            "cuboctahedral CN_12",
            "q2 CN_12",
            "q4 CN_12",
            "q6 CN_12",
        ]
        for i in range(len(l)):
            self.assertEqual(l[i], t[i])
        ops = opsf.featurize(self.sc, 0)
        self.assertEqual(len(ops), 37)
        self.assertAlmostEqual(ops[opsf.feature_labels().index("octahedral CN_6")], 0.9995, places=7)
        ops = opsf.featurize(self.cscl, 0)
        self.assertAlmostEqual(
            ops[opsf.feature_labels().index("body-centered cubic CN_8")],
            0.8955,
            places=7,
        )
        opsf = OPSiteFingerprint(dist_exp=0)
        ops = opsf.featurize(self.cscl, 0)
        self.assertAlmostEqual(
            ops[opsf.feature_labels().index("body-centered cubic CN_8")],
            0.9555,
            places=7,
        )

        # The following test aims at ensuring the copying of the OP dictionaries work.
        opsfp = OPSiteFingerprint()
        cnnfp = CrystalNNFingerprint.from_preset("ops")
        self.assertEqual(len([1 for l in opsfp.feature_labels() if l.split()[0] == "wt"]), 0)

    def test_crystal_nn_fingerprint(self):
        cnnfp = CrystalNNFingerprint.from_preset("ops", distance_cutoffs=None, x_diff_weight=None)
        l = cnnfp.feature_labels()
        t = [
            "wt CN_1",
            "sgl_bd CN_1",
            "wt CN_2",
            "L-shaped CN_2",
            "water-like CN_2",
            "bent 120 degrees CN_2",
            "bent 150 degrees CN_2",
            "linear CN_2",
            "wt CN_3",
            "trigonal planar CN_3",
            "trigonal non-coplanar CN_3",
            "T-shaped CN_3",
            "wt CN_4",
            "square co-planar CN_4",
            "tetrahedral CN_4",
            "rectangular see-saw-like CN_4",
            "see-saw-like CN_4",
            "trigonal pyramidal CN_4",
            "wt CN_5",
            "pentagonal planar CN_5",
            "square pyramidal CN_5",
            "trigonal bipyramidal CN_5",
            "wt CN_6",
            "hexagonal planar CN_6",
            "octahedral CN_6",
            "pentagonal pyramidal CN_6",
            "wt CN_7",
            "hexagonal pyramidal CN_7",
            "pentagonal bipyramidal CN_7",
            "wt CN_8",
            "body-centered cubic CN_8",
            "hexagonal bipyramidal CN_8",
            "wt CN_9",
            "q2 CN_9",
            "q4 CN_9",
            "q6 CN_9",
            "wt CN_10",
            "q2 CN_10",
            "q4 CN_10",
            "q6 CN_10",
            "wt CN_11",
            "q2 CN_11",
            "q4 CN_11",
            "q6 CN_11",
            "wt CN_12",
            "cuboctahedral CN_12",
            "q2 CN_12",
            "q4 CN_12",
            "q6 CN_12",
            "wt CN_13",
            "wt CN_14",
            "wt CN_15",
            "wt CN_16",
            "wt CN_17",
            "wt CN_18",
            "wt CN_19",
            "wt CN_20",
            "wt CN_21",
            "wt CN_22",
            "wt CN_23",
            "wt CN_24",
        ]
        for i in range(len(l)):
            self.assertEqual(l[i], t[i])
        ops = cnnfp.featurize(self.sc, 0)
        self.assertEqual(len(ops), 61)
        self.assertAlmostEqual(ops[cnnfp.feature_labels().index("wt CN_6")], 1, places=7)
        self.assertAlmostEqual(ops[cnnfp.feature_labels().index("octahedral CN_6")], 1, places=7)
        ops = cnnfp.featurize(self.cscl, 0)
        self.assertAlmostEqual(ops[cnnfp.feature_labels().index("wt CN_8")], 0.498099, places=3)

        self.assertAlmostEqual(
            ops[cnnfp.feature_labels().index("body-centered cubic CN_8")],
            0.47611,
            places=3,
        )

        op_types = {6: ["wt", "oct_max"], 8: ["wt", "bcc"]}
        cnnfp = CrystalNNFingerprint(op_types, distance_cutoffs=None, x_diff_weight=None)
        labels = ["wt CN_6", "oct_max CN_6", "wt CN_8", "bcc CN_8"]
        for l1, l2 in zip(cnnfp.feature_labels(), labels):
            self.assertEqual(l1, l2)
        feats = cnnfp.featurize(self.sc, 0)
        self.assertEqual(len(feats), 4)

        chem_info = {
            "mass": {"Al": 26.9, "Cs+": 132.9, "Cl-": 35.4},
            "Pauling scale": {"Al": 1.61, "Cs+": 0.79, "Cl-": 3.16},
        }
        cnnchemfp = CrystalNNFingerprint(op_types, chem_info=chem_info, distance_cutoffs=None, x_diff_weight=None)
        labels = labels + ["mass local diff", "Pauling scale local diff"]
        for l1, l2 in zip(cnnchemfp.feature_labels(), labels):
            self.assertEqual(l1, l2)

        feats = cnnchemfp.featurize(self.sc, 0)
        self.assertEqual(len(feats), 6)
        self.assertAlmostEqual(feats[cnnchemfp.feature_labels().index("wt CN_6")], 1, places=7)
        self.assertAlmostEqual(feats[cnnchemfp.feature_labels().index("oct_max CN_6")], 1, places=7)
        self.assertAlmostEqual(feats[cnnchemfp.feature_labels().index("mass local diff")], 0, places=7)
        self.assertAlmostEqual(
            feats[cnnchemfp.feature_labels().index("Pauling scale local diff")],
            0,
            places=7,
        )

        feats = cnnchemfp.featurize(self.cscl, 0)
        self.assertAlmostEqual(feats[cnnchemfp.feature_labels().index("bcc CN_8")], 0.4761107, places=3)
        self.assertAlmostEqual(feats[cnnchemfp.feature_labels().index("mass local diff")], 97.5, places=3)
        self.assertAlmostEqual(
            feats[cnnchemfp.feature_labels().index("Pauling scale local diff")],
            -2.37,
            places=3,
        )

    def test_chemenv_site_fingerprint(self):
        cefp = ChemEnvSiteFingerprint.from_preset("multi_weights")
        l = cefp.feature_labels()
        cevals = cefp.featurize(self.sc, 0)
        self.assertEqual(len(cevals), 66)
        self.assertAlmostEqual(cevals[l.index("O:6")], 1, places=7)
        self.assertAlmostEqual(cevals[l.index("C:8")], 0, places=7)
        cevals = cefp.featurize(self.cscl, 0)
        self.assertAlmostEqual(cevals[l.index("C:8")], 0.9953721, places=7)
        self.assertAlmostEqual(cevals[l.index("O:6")], 0, places=7)
        cefp = ChemEnvSiteFingerprint.from_preset("simple")
        l = cefp.feature_labels()
        cevals = cefp.featurize(self.sc, 0)
        self.assertEqual(len(cevals), 66)
        self.assertAlmostEqual(cevals[l.index("O:6")], 1, places=7)
        self.assertAlmostEqual(cevals[l.index("C:8")], 0, places=7)
        cevals = cefp.featurize(self.cscl, 0)
        self.assertAlmostEqual(cevals[l.index("C:8")], 0.9953721, places=7)
        self.assertAlmostEqual(cevals[l.index("O:6")], 0, places=7)

    def test_voronoifingerprint(self):
        df_sc = pd.DataFrame({"struct": [self.sc], "site": [0]})
        vorofp = VoronoiFingerprint(use_symm_weights=True)
        vorofps = vorofp.featurize_dataframe(df_sc, ["struct", "site"])
        self.assertAlmostEqual(vorofps["Voro_index_3"][0], 0.0)
        self.assertAlmostEqual(vorofps["Voro_index_4"][0], 6.0)
        self.assertAlmostEqual(vorofps["Voro_index_5"][0], 0.0)
        self.assertAlmostEqual(vorofps["Voro_index_6"][0], 0.0)
        self.assertAlmostEqual(vorofps["Voro_index_7"][0], 0.0)
        self.assertAlmostEqual(vorofps["Voro_index_8"][0], 0.0)
        self.assertAlmostEqual(vorofps["Voro_index_9"][0], 0.0)
        self.assertAlmostEqual(vorofps["Voro_index_10"][0], 0.0)
        self.assertAlmostEqual(vorofps["Symmetry_index_3"][0], 0.0)
        self.assertAlmostEqual(vorofps["Symmetry_index_4"][0], 1.0)
        self.assertAlmostEqual(vorofps["Symmetry_index_5"][0], 0.0)
        self.assertAlmostEqual(vorofps["Symmetry_index_6"][0], 0.0)
        self.assertAlmostEqual(vorofps["Symmetry_index_7"][0], 0.0)
        self.assertAlmostEqual(vorofps["Symmetry_index_8"][0], 0.0)
        self.assertAlmostEqual(vorofps["Symmetry_index_9"][0], 0.0)
        self.assertAlmostEqual(vorofps["Symmetry_index_10"][0], 0.0)
        self.assertAlmostEqual(vorofps["Symmetry_weighted_index_3"][0], 0.0)
        self.assertAlmostEqual(vorofps["Symmetry_weighted_index_4"][0], 1.0)
        self.assertAlmostEqual(vorofps["Symmetry_weighted_index_5"][0], 0.0)
        self.assertAlmostEqual(vorofps["Symmetry_weighted_index_6"][0], 0.0)
        self.assertAlmostEqual(vorofps["Symmetry_weighted_index_7"][0], 0.0)
        self.assertAlmostEqual(vorofps["Symmetry_weighted_index_8"][0], 0.0)
        self.assertAlmostEqual(vorofps["Symmetry_weighted_index_9"][0], 0.0)
        self.assertAlmostEqual(vorofps["Symmetry_weighted_index_10"][0], 0.0)
        self.assertAlmostEqual(vorofps["Voro_vol_sum"][0], 43.614208)
        self.assertAlmostEqual(vorofps["Voro_area_sum"][0], 74.3424)
        self.assertAlmostEqual(vorofps["Voro_vol_mean"][0], 7.269034667)
        self.assertAlmostEqual(vorofps["Voro_vol_std_dev"][0], 0.0)
        self.assertAlmostEqual(vorofps["Voro_vol_minimum"][0], 7.269034667)
        self.assertAlmostEqual(vorofps["Voro_vol_maximum"][0], 7.269034667)
        self.assertAlmostEqual(vorofps["Voro_area_mean"][0], 12.3904)
        self.assertAlmostEqual(vorofps["Voro_area_std_dev"][0], 0.0)
        self.assertAlmostEqual(vorofps["Voro_area_minimum"][0], 12.3904)
        self.assertAlmostEqual(vorofps["Voro_area_maximum"][0], 12.3904)
        self.assertAlmostEqual(vorofps["Voro_dist_mean"][0], 3.52)
        self.assertAlmostEqual(vorofps["Voro_dist_std_dev"][0], 0.0)
        self.assertAlmostEqual(vorofps["Voro_dist_minimum"][0], 3.52)
        self.assertAlmostEqual(vorofps["Voro_dist_maximum"][0], 3.52)


if __name__ == "__main__":
    unittest.main()
