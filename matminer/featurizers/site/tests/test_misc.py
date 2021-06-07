import unittest

import pandas as pd
from pymatgen.core import Structure, Lattice
from pymatgen.analysis.local_env import VoronoiNN, JmolNN, CrystalNN

from matminer.featurizers.site.misc import (
    IntersticeDistribution,
    CoordinationNumber,
)
from matminer.featurizers.site.tests.base import SiteFeaturizerTest


class MiscSiteTests(SiteFeaturizerTest):
    def test_interstice_distribution_of_crystal(self):
        bcc_li = Structure(
            Lattice([[3.51, 0, 0], [0, 3.51, 0], [0, 0, 3.51]]),
            ["Li"] * 2,
            [[0, 0, 0], [0.5, 0.5, 0.5]],
        )
        df_bcc_li = pd.DataFrame({"struct": [bcc_li], "site": [1]})

        interstice_distribution = IntersticeDistribution()
        intersticefp = interstice_distribution.featurize_dataframe(df_bcc_li, ["struct", "site"])

        self.assertAlmostEqual(intersticefp["Interstice_vol_mean"][0], 0.32, 2)
        self.assertAlmostEqual(intersticefp["Interstice_vol_std_dev"][0], 0)
        self.assertAlmostEqual(intersticefp["Interstice_vol_minimum"][0], 0.32, 2)
        self.assertAlmostEqual(intersticefp["Interstice_vol_maximum"][0], 0.32, 2)
        self.assertAlmostEqual(intersticefp["Interstice_area_mean"][0], 0.16682, 5)
        self.assertAlmostEqual(intersticefp["Interstice_area_std_dev"][0], 0)
        self.assertAlmostEqual(intersticefp["Interstice_area_minimum"][0], 0.16682, 5)
        self.assertAlmostEqual(intersticefp["Interstice_area_maximum"][0], 0.16682, 5)
        self.assertAlmostEqual(intersticefp["Interstice_dist_mean"][0], 0.06621, 5)
        self.assertAlmostEqual(intersticefp["Interstice_dist_std_dev"][0], 0.07655, 5)
        self.assertAlmostEqual(intersticefp["Interstice_dist_minimum"][0], 0, 3)
        self.assertAlmostEqual(intersticefp["Interstice_dist_maximum"][0], 0.15461, 5)

    def test_interstice_distribution_of_glass(self):
        cuzr_glass = Structure(
            Lattice([[25, 0, 0], [0, 25, 0], [0, 0, 25]]),
            [
                "Cu",
                "Cu",
                "Cu",
                "Cu",
                "Cu",
                "Zr",
                "Cu",
                "Zr",
                "Cu",
                "Zr",
                "Cu",
                "Zr",
                "Cu",
                "Cu",
            ],
            [
                [11.81159679, 16.49480537, 21.69139442],
                [11.16777208, 17.87850033, 18.57877144],
                [12.22394796, 15.83218325, 19.37763412],
                [13.07053548, 14.34025424, 21.77557646],
                [10.78147725, 19.61647494, 20.77595531],
                [10.87541011, 14.65986432, 23.61517624],
                [12.76631002, 18.41479521, 20.46717947],
                [14.63911675, 16.47487037, 20.52671362],
                [14.2470256, 18.44215167, 22.56257566],
                [9.38050168, 16.87974592, 20.51885879],
                [10.66332986, 14.43900833, 20.545186],
                [11.57096832, 18.79848982, 23.26073408],
                [13.27048138, 16.38613795, 23.59697472],
                [9.55774984, 17.09220537, 23.1856528],
            ],
            coords_are_cartesian=True,
        )
        df_glass = pd.DataFrame({"struct": [cuzr_glass], "site": [0]})

        interstice_distribution = IntersticeDistribution()
        intersticefp = interstice_distribution.featurize_dataframe(df_glass, ["struct", "site"])

        self.assertAlmostEqual(intersticefp["Interstice_vol_mean"][0], 0.28905, 5)
        self.assertAlmostEqual(intersticefp["Interstice_vol_std_dev"][0], 0.04037, 5)
        self.assertAlmostEqual(intersticefp["Interstice_vol_minimum"][0], 0.21672, 5)
        self.assertAlmostEqual(intersticefp["Interstice_vol_maximum"][0], 0.39084, 5)
        self.assertAlmostEqual(intersticefp["Interstice_area_mean"][0], 0.16070, 5)
        self.assertAlmostEqual(intersticefp["Interstice_area_std_dev"][0], 0.05245, 5)
        self.assertAlmostEqual(intersticefp["Interstice_area_minimum"][0], 0.07132, 5)
        self.assertAlmostEqual(intersticefp["Interstice_area_maximum"][0], 0.26953, 5)
        self.assertAlmostEqual(intersticefp["Interstice_dist_mean"][0], 0.08154, 5)
        self.assertAlmostEqual(intersticefp["Interstice_dist_std_dev"][0], 0.14778, 5)
        self.assertAlmostEqual(intersticefp["Interstice_dist_minimum"][0], -0.04668, 5)
        self.assertAlmostEqual(intersticefp["Interstice_dist_maximum"][0], 0.37565, 5)

    def test_cns(self):
        cnv = CoordinationNumber.from_preset("VoronoiNN")
        self.assertEqual(len(cnv.feature_labels()), 1)
        self.assertEqual(cnv.feature_labels()[0], "CN_VoronoiNN")
        self.assertAlmostEqual(cnv.featurize(self.sc, 0)[0], 6)
        self.assertAlmostEqual(cnv.featurize(self.cscl, 0)[0], 14)
        self.assertAlmostEqual(cnv.featurize(self.cscl, 1)[0], 14)
        self.assertEqual(len(cnv.citations()), 2)
        cnv = CoordinationNumber(VoronoiNN(), use_weights="sum")
        self.assertEqual(cnv.feature_labels()[0], "CN_VoronoiNN")
        self.assertAlmostEqual(cnv.featurize(self.cscl, 0)[0], 9.2584516)
        self.assertAlmostEqual(cnv.featurize(self.cscl, 1)[0], 9.2584516)
        self.assertEqual(len(cnv.citations()), 2)
        cnv = CoordinationNumber(VoronoiNN(), use_weights="effective")
        self.assertEqual(cnv.feature_labels()[0], "CN_VoronoiNN")
        self.assertAlmostEqual(cnv.featurize(self.cscl, 0)[0], 11.648923254)
        self.assertAlmostEqual(cnv.featurize(self.cscl, 1)[0], 11.648923254)
        self.assertEqual(len(cnv.citations()), 2)
        cnj = CoordinationNumber.from_preset("JmolNN")
        self.assertEqual(cnj.feature_labels()[0], "CN_JmolNN")
        self.assertAlmostEqual(cnj.featurize(self.sc, 0)[0], 0)
        self.assertAlmostEqual(cnj.featurize(self.cscl, 0)[0], 0)
        self.assertAlmostEqual(cnj.featurize(self.cscl, 1)[0], 0)
        self.assertEqual(len(cnj.citations()), 1)
        jmnn = JmolNN(el_radius_updates={"Al": 1.55, "Cl": 1.7, "Cs": 1.7})
        cnj = CoordinationNumber(jmnn)
        self.assertEqual(cnj.feature_labels()[0], "CN_JmolNN")
        self.assertAlmostEqual(cnj.featurize(self.sc, 0)[0], 6)
        self.assertAlmostEqual(cnj.featurize(self.cscl, 0)[0], 8)
        self.assertAlmostEqual(cnj.featurize(self.cscl, 1)[0], 8)
        self.assertEqual(len(cnj.citations()), 1)
        cnmd = CoordinationNumber.from_preset("MinimumDistanceNN")
        self.assertEqual(cnmd.feature_labels()[0], "CN_MinimumDistanceNN")
        self.assertAlmostEqual(cnmd.featurize(self.sc, 0)[0], 6)
        self.assertAlmostEqual(cnmd.featurize(self.cscl, 0)[0], 8)
        self.assertAlmostEqual(cnmd.featurize(self.cscl, 1)[0], 8)
        self.assertEqual(len(cnmd.citations()), 1)
        cnmok = CoordinationNumber.from_preset("MinimumOKeeffeNN")
        self.assertEqual(cnmok.feature_labels()[0], "CN_MinimumOKeeffeNN")
        self.assertAlmostEqual(cnmok.featurize(self.sc, 0)[0], 6)
        self.assertAlmostEqual(cnmok.featurize(self.cscl, 0)[0], 8)
        self.assertAlmostEqual(cnmok.featurize(self.cscl, 1)[0], 6)
        self.assertEqual(len(cnmok.citations()), 2)
        cnmvire = CoordinationNumber.from_preset("MinimumVIRENN")
        self.assertEqual(cnmvire.feature_labels()[0], "CN_MinimumVIRENN")
        self.assertAlmostEqual(cnmvire.featurize(self.sc, 0)[0], 6)
        self.assertAlmostEqual(cnmvire.featurize(self.cscl, 0)[0], 8)
        self.assertAlmostEqual(cnmvire.featurize(self.cscl, 1)[0], 14)
        self.assertEqual(len(cnmvire.citations()), 2)
        self.assertEqual(len(cnmvire.implementors()), 2)
        self.assertEqual(cnmvire.implementors()[0], "Nils E. R. Zimmermann")


if __name__ == "__main__":
    unittest.main()
