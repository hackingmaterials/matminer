import math
import unittest
from unittest import SkipTest

import pandas as pd
import numpy as np
from pymatgen.core import Composition
from pymatgen.ext.matproj import MPRester
from pymatgen.core.periodic_table import Specie, Element

from matminer.featurizers.composition.tests.base import CompositionFeaturesTest
from matminer.featurizers.conversions import CompositionToOxidComposition
from matminer.featurizers.composition.alloy import YangSolidSolution, Miedema, WenAlloys


class AlloyFeaturizersTest(CompositionFeaturesTest):
    def test_miedema_all(self):
        df = pd.DataFrame(
            {
                "composition": [
                    Composition("TiZr"),
                    Composition("Mg10Cu50Ca40"),
                    Composition("Fe2O3"),
                ]
            }
        )
        miedema = Miedema(struct_types="all")
        self.assertTrue(miedema.precheck(df["composition"].iloc[0]))
        self.assertFalse(miedema.precheck(df["composition"].iloc[-1]))
        self.assertAlmostEqual(miedema.precheck_dataframe(df, "composition"), 2 / 3)

        # test precheck for oxidation-state decorated compositions
        df = CompositionToOxidComposition(return_original_on_error=True).featurize_dataframe(df, "composition")
        self.assertTrue(miedema.precheck(df["composition_oxid"].iloc[0]))
        self.assertFalse(miedema.precheck(df["composition_oxid"].iloc[-1]))
        self.assertAlmostEqual(miedema.precheck_dataframe(df, "composition_oxid"), 2 / 3)

        mfps = miedema.featurize_dataframe(df, col_id="composition")
        self.assertAlmostEqual(mfps["Miedema_deltaH_inter"][0], -0.003445022152)
        self.assertAlmostEqual(mfps["Miedema_deltaH_amor"][0], 0.0707658836300)
        self.assertAlmostEqual(mfps["Miedema_deltaH_ss_min"][0], 0.03663599755)

        self.assertAlmostEqual(mfps["Miedema_deltaH_inter"][1], -0.235125978427)
        self.assertAlmostEqual(mfps["Miedema_deltaH_amor"][1], -0.164541848271)
        self.assertAlmostEqual(mfps["Miedema_deltaH_ss_min"][1], -0.05280843311)

        self.assertAlmostEqual(math.isnan(mfps["Miedema_deltaH_inter"][2]), True)
        self.assertAlmostEqual(math.isnan(mfps["Miedema_deltaH_amor"][2]), True)
        self.assertAlmostEqual(math.isnan(mfps["Miedema_deltaH_ss_min"][2]), True)

        # make sure featurization works equally for compositions with or without
        # oxidation states
        mfps = miedema.featurize_dataframe(df, col_id="composition_oxid")
        self.assertAlmostEqual(mfps["Miedema_deltaH_inter"][0], -0.003445022152)
        self.assertAlmostEqual(mfps["Miedema_deltaH_amor"][0], 0.0707658836300)
        self.assertAlmostEqual(mfps["Miedema_deltaH_ss_min"][0], 0.03663599755)

    def test_miedema_ss(self):
        df = pd.DataFrame(
            {
                "composition": [
                    Composition("TiZr"),
                    Composition("Mg10Cu50Ca40"),
                    Composition("Fe2O3"),
                ]
            }
        )
        miedema = Miedema(struct_types="ss", ss_types=["min", "fcc", "bcc", "hcp", "no_latt"])
        mfps = miedema.featurize_dataframe(df, col_id="composition")
        self.assertAlmostEqual(mfps["Miedema_deltaH_ss_min"][0], 0.03663599755)
        self.assertAlmostEqual(mfps["Miedema_deltaH_ss_fcc"][0], 0.04700027066)
        self.assertAlmostEqual(mfps["Miedema_deltaH_ss_bcc"][0], 0.08327522653)
        self.assertAlmostEqual(mfps["Miedema_deltaH_ss_hcp"][0], 0.03663599755)
        self.assertAlmostEqual(mfps["Miedema_deltaH_ss_no_latt"][0], 0.036635998)

        self.assertAlmostEqual(mfps["Miedema_deltaH_ss_min"][1], -0.05280843311)
        self.assertAlmostEqual(mfps["Miedema_deltaH_ss_fcc"][1], 0.03010575174)
        self.assertAlmostEqual(mfps["Miedema_deltaH_ss_bcc"][1], -0.05280843311)
        self.assertAlmostEqual(mfps["Miedema_deltaH_ss_hcp"][1], 0.03010575174)
        self.assertAlmostEqual(mfps["Miedema_deltaH_ss_no_latt"][1], -0.0035781359)

        self.assertAlmostEqual(math.isnan(mfps["Miedema_deltaH_ss_min"][2]), True)
        self.assertAlmostEqual(math.isnan(mfps["Miedema_deltaH_ss_fcc"][2]), True)
        self.assertAlmostEqual(math.isnan(mfps["Miedema_deltaH_ss_bcc"][2]), True)
        self.assertAlmostEqual(math.isnan(mfps["Miedema_deltaH_ss_hcp"][2]), True)
        self.assertAlmostEqual(math.isnan(mfps["Miedema_deltaH_ss_no_latt"][2]), True)

    def test_yang(self):
        comps = list(
            map(
                Composition,
                ["ZrHfTiCuNi", "CuNi", "CoCrFeNiCuAl0.3", "CoCrFeNiCuAl", "LaO3"],
            )
        )

        # Run the featurization
        feat = YangSolidSolution()

        df = pd.DataFrame({"composition": comps})
        self.assertFalse(feat.precheck(df["composition"].iloc[-1]))
        self.assertAlmostEqual(feat.precheck_dataframe(df, "composition"), 0.8, places=2)

        # test precheck for oxidation-state decorated compositions
        df = CompositionToOxidComposition(return_original_on_error=True).featurize_dataframe(df, "composition")
        self.assertFalse(feat.precheck(df["composition_oxid"].iloc[-1]))
        self.assertAlmostEqual(feat.precheck_dataframe(df, "composition_oxid"), 0.8, places=2)

        feat.set_n_jobs(1)
        features = feat.featurize_many(comps)

        # Check the results
        #  These are compared to results from the original paper,
        #   except for CoCrFeNiCuAl0.3, where the paper reports a value
        #   exactly 1/10th of what I compute using Excel and matminer
        # I use a high tolerance because matminer uses a different source
        #   of radii than the original paper (do not have Kittel's atomic
        #   radii available)
        self.assertEqual((5, 2), np.array(features).shape)
        self.assertArrayAlmostEqual([0.95, 0.1021], features[0], decimal=2)
        self.assertArrayAlmostEqual([2.22, 0.0], features[1], decimal=2)
        self.assertArrayAlmostEqual([158.5, 0.0315], features[2], decimal=1)
        self.assertArrayAlmostEqual([5.06, 0.0482], features[3], decimal=1)

    def test_WenAlloys(self):
        wa = WenAlloys()
        c1 = "Fe0.62C0.000953Mn0.000521Si0.00102Cr0.00011Ni0.192" "Mo0.0176V0.000112Nb6.16e-05Co0.146Al0.00318Ti0.0185"
        c2 = (
            "Fe0.623C0.00854Mn0.000104Si0.000203Cr0.147Ni9.71e-05"
            "Mo0.0179V0.00515N0.00163Nb6.14e-05Co0.188W0.00729Al0.000845"
        )
        comp = Composition(c1)

        # Test prechecking
        comp_bad = Composition("LaO3")
        self.assertTrue(wa.precheck(comp))
        self.assertFalse(wa.precheck(comp_bad))

        f = wa.featurize(comp)

        d = dict(zip(wa.feature_labels(), f))
        correct = {
            "APE mean": 0.018915555593392162,
            "Atomic Fraction": "Fe0.6199642900568927C0.0009529451103616431Mn0.0005209699921284533Si"
            "0.0010199412513839203Cr0.00010999366436493258Ni0.1919889414369732Mo"
            "0.017598986298389213V0.0001119935491715677Nb6.159645204436224e-05Co"
            "0.14599159088436503Al0.003179816842549869Ti0.018498934461375023",
            "Atomic weight mean": 57.24008321450784,
            "Configuration entropy": -0.008958911485121818,
            "Electronegativity delta": 0.042327487126447516,
            "Electronegativity local mismatch": 0.08262466022141576,
            "Interant d electrons": 45.0,
            "Interant electrons": 53.0,
            "Interant f electrons": 0,
            "Interant p electrons": 5.0,
            "Interant s electrons": 3.0,
            "Lambda entropy": -12.084431980055149,
            "Mean cohesive energy": 4.382084353941212,
            "Mixing enthalpy": 3.6650695863166347,
            "Radii gamma": 1.4183511064895242,
            "Radii local mismatch": 0.7953797741513383,
            "Shear modulus delta": 0.1794147729878139,
            "Shear modulus local mismatch": 3.192861083726266,
            "Shear modulus mean": 79.48600137832061,
            "Shear modulus strength model": -0.009636621848440554,
            "Total weight": 57.243028243301005,
            "VEC mean": 8.447250438374748,
            "Weight Fraction": "Fe0.6048579375087819 C0.00019995792415715736 "
            "Mn0.0005000210911858884 Si0.0005004488909678273 "
            "Cr9.991733798026916e-05 Ni0.19686472127404955 "
            "Mo0.029497810507563525 V9.967061797901463e-05 "
            "Nb9.997781710071831e-05 Co0.15031081922202344 "
            "Al0.0014988950686416751 Ti0.015469822739568852 ",
            "Yang delta": 0.027227922269552986,
            "Yang omega": 4.4226005659658725,
        }

        for flabel, fvalue in d.items():
            correct_value = correct[flabel]
            if isinstance(correct_value, str):
                self.assertEqual(correct_value, fvalue)
            else:
                self.assertAlmostEqual(correct_value, fvalue, places=8)

        self.assertEqual(len(wa.feature_labels()), 25)

        df = pd.DataFrame({"composition": [comp, Composition(c2)]})

        df = wa.featurize_dataframe(df, "composition")
        self.assertTupleEqual(df.shape, (2, 26))
        self.assertAlmostEqual(df["Configuration entropy"].iloc[0], -0.008959, places=5)
        self.assertAlmostEqual(df["Configuration entropy"].iloc[1], -0.009039, places=5)


if __name__ == "__main__":
    unittest.main()
