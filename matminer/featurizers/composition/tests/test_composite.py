import math
import unittest

from matminer.featurizers.composition.composite import ElementProperty, Meredig
from matminer.featurizers.composition.tests.base import CompositionFeaturesTest


class CompositeFeaturesTest(CompositionFeaturesTest):
    def test_elem(self):
        df_elem = ElementProperty.from_preset("magpie").featurize_dataframe(self.df, col_id="composition")
        self.assertAlmostEqual(df_elem["MagpieData minimum Number"][0], 8)
        self.assertAlmostEqual(df_elem["MagpieData maximum Number"][0], 26)
        self.assertAlmostEqual(df_elem["MagpieData range Number"][0], 18)
        self.assertAlmostEqual(df_elem["MagpieData mean Number"][0], 15.2)
        self.assertAlmostEqual(df_elem["MagpieData avg_dev Number"][0], 8.64)
        self.assertAlmostEqual(df_elem["MagpieData mode Number"][0], 8)

    def test_elem_deml(self):
        df_elem_deml = ElementProperty.from_preset("deml").featurize_dataframe(self.df, col_id="composition")
        self.assertAlmostEqual(df_elem_deml["DemlData minimum atom_num"][0], 8)
        self.assertAlmostEqual(df_elem_deml["DemlData maximum atom_num"][0], 26)
        self.assertAlmostEqual(df_elem_deml["DemlData range atom_num"][0], 18)
        self.assertAlmostEqual(df_elem_deml["DemlData mean atom_num"][0], 15.2)
        self.assertAlmostEqual(df_elem_deml["DemlData std_dev atom_num"][0], 12.7279, 4)

    def test_elem_matminer(self):
        df_elem = ElementProperty.from_preset("matminer").featurize_dataframe(self.df, col_id="composition")
        self.assertAlmostEqual(df_elem["PymatgenData minimum melting_point"][0], 54.8, 1)
        self.assertTrue(math.isnan(df_elem["PymatgenData maximum bulk_modulus"][0]))
        self.assertAlmostEqual(df_elem["PymatgenData range X"][0], 1.61, 1)
        self.assertAlmostEqual(df_elem["PymatgenData mean X"][0], 2.796, 1)
        self.assertAlmostEqual(df_elem["PymatgenData maximum block"][0], 3, 1)

    def test_elem_matscholar_el(self):
        df_elem = ElementProperty.from_preset("matscholar_el").featurize_dataframe(self.df, col_id="composition")
        self.assertAlmostEqual(
            df_elem["MatscholarElementData range embedding 149"].iloc[0],
            0.06827970966696739,
        )
        self.assertAlmostEqual(
            df_elem["MatscholarElementData range embedding 149"].iloc[1],
            0.06827970966696739,
        )
        self.assertAlmostEqual(
            df_elem["MatscholarElementData mean embedding 18"].iloc[0],
            -0.020534400502219795,
        )
        self.assertAlmostEqual(
            df_elem["MatscholarElementData mean embedding 18"].iloc[1],
            -0.02483355056028813,
        )

    def test_elem_megnet_el(self):
        ep = ElementProperty.from_preset("megnet_el")
        df_elem = ep.featurize_dataframe(self.df, col_id="composition")
        self.assertAlmostEqual(df_elem["MEGNetElementData maximum embedding 1"].iloc[0], 0.127333, places=6)
        self.assertAlmostEqual(df_elem["MEGNetElementData maximum embedding 1"].iloc[1], 0.127333, places=6)
        self.assertAlmostEqual(
            df_elem["MEGNetElementData maximum embedding 11"].iloc[0],
            0.160505,
            places=6,
        )
        self.assertAlmostEqual(
            df_elem["MEGNetElementData maximum embedding 11"].iloc[1],
            0.160505,
            places=6,
        )
        self.assertTrue(ep.citations())

    def test_meredig(self):
        df_val = Meredig().featurize_dataframe(self.df, col_id="composition")
        self.assertAlmostEqual(df_val["Fe fraction"].iloc[0], 2.0 / 5.0)
        self.assertAlmostEqual(df_val["Fe fraction"].iloc[1], 0.5)
        self.assertAlmostEqual(df_val["O fraction"].iloc[0], 3.0 / 5.0)
        self.assertAlmostEqual(df_val["O fraction"].iloc[1], 0.5)
        self.assertAlmostEqual(df_val["frac s valence electrons"].iloc[0], 0.294117647)
        self.assertAlmostEqual(df_val["mean Number"].iloc[0], 15.2)

    def test_fere_corr(self):
        df_fere_corr = ElementProperty(
            features=["FERE correction"],
            stats=["minimum", "maximum", "range", "mean", "std_dev"],
            data_source="deml",
        ).featurize_dataframe(self.df, col_id="composition")
        self.assertAlmostEqual(df_fere_corr["DemlData minimum FERE correction"][0], -0.15213431610903)
        self.assertAlmostEqual(df_fere_corr["DemlData maximum FERE correction"][0], 0.23)
        self.assertAlmostEqual(df_fere_corr["DemlData range FERE correction"][0], 0.382134316)
        self.assertAlmostEqual(df_fere_corr["DemlData mean FERE correction"][0], 0.077146274)
        self.assertAlmostEqual(df_fere_corr["DemlData std_dev FERE correction"][0], 0.270209766)


if __name__ == "__main__":
    unittest.main()
