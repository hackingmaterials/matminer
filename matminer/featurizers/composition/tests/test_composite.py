import math
import unittest

from matminer.featurizers.composition.composite import ElementProperty, Meredig
from matminer.featurizers.composition.tests.base import CompositionFeaturesTest


class CompositeFeaturesTest(CompositionFeaturesTest):
    def test_elem(self):
        df_elem = ElementProperty.from_preset("magpie", impute_nan=False).featurize_dataframe(
            self.df, col_id="composition"
        )
        self.assertAlmostEqual(df_elem["MagpieData minimum Number"][0], 8)
        self.assertAlmostEqual(df_elem["MagpieData maximum Number"][0], 26)
        self.assertAlmostEqual(df_elem["MagpieData range Number"][0], 18)
        self.assertAlmostEqual(df_elem["MagpieData mean Number"][0], 15.2)
        self.assertAlmostEqual(df_elem["MagpieData avg_dev Number"][0], 8.64)
        self.assertAlmostEqual(df_elem["MagpieData mode Number"][0], 8)

        df_elem = ElementProperty.from_preset("magpie", impute_nan=False).featurize_dataframe(
            self.df_nans, col_id="composition"
        )
        self.assertEqual(df_elem.isna().sum().sum(), 132)

        df_elem = ElementProperty.from_preset("magpie", impute_nan=True).featurize_dataframe(
            self.df, col_id="composition"
        )
        self.assertAlmostEqual(df_elem["MagpieData minimum Number"][0], 8)
        self.assertAlmostEqual(df_elem["MagpieData maximum Number"][0], 26)
        self.assertAlmostEqual(df_elem["MagpieData range Number"][0], 18)
        self.assertAlmostEqual(df_elem["MagpieData mean Number"][0], 15.2)
        self.assertAlmostEqual(df_elem["MagpieData avg_dev Number"][0], 8.64)
        self.assertAlmostEqual(df_elem["MagpieData mode Number"][0], 8)
        self.assertEqual(df_elem.isna().sum().sum(), 0)

        df_elem = ElementProperty.from_preset("magpie", impute_nan=True).featurize_dataframe(
            self.df_nans, col_id="composition"
        )
        self.assertAlmostEqual(df_elem["MagpieData minimum Number"][0], 56.5)
        self.assertAlmostEqual(df_elem["MagpieData maximum Number"][0], 92)
        self.assertAlmostEqual(df_elem["MagpieData range Number"][0], 35.5)
        self.assertAlmostEqual(df_elem["MagpieData mean Number"][0], 70.7)
        self.assertAlmostEqual(df_elem["MagpieData avg_dev Number"][0], 17.04)
        self.assertAlmostEqual(df_elem["MagpieData mode Number"][0], 56.5)
        self.assertEqual(df_elem.isna().sum().sum(), 0)

    def test_elem_deml(self):
        df_elem_deml = ElementProperty.from_preset("deml", impute_nan=False).featurize_dataframe(
            self.df, col_id="composition"
        )
        self.assertAlmostEqual(df_elem_deml["DemlData minimum atom_num"][0], 8)
        self.assertAlmostEqual(df_elem_deml["DemlData maximum atom_num"][0], 26)
        self.assertAlmostEqual(df_elem_deml["DemlData range atom_num"][0], 18)
        self.assertAlmostEqual(df_elem_deml["DemlData mean atom_num"][0], 15.2)
        self.assertAlmostEqual(df_elem_deml["DemlData std_dev atom_num"][0], 12.7279, 4)

        df_elem_deml = ElementProperty.from_preset("deml", impute_nan=False).featurize_dataframe(
            self.df_nans, col_id="composition", ignore_errors=True
        )
        self.assertEqual(df_elem_deml.isna().sum().sum(), 80)

        df_elem_deml = ElementProperty.from_preset("deml", impute_nan=True).featurize_dataframe(
            self.df, col_id="composition"
        )
        self.assertAlmostEqual(df_elem_deml["DemlData minimum atom_num"][0], 8)
        self.assertAlmostEqual(df_elem_deml["DemlData maximum atom_num"][0], 26)
        self.assertAlmostEqual(df_elem_deml["DemlData range atom_num"][0], 18)
        self.assertAlmostEqual(df_elem_deml["DemlData mean atom_num"][0], 15.2)
        self.assertAlmostEqual(df_elem_deml["DemlData std_dev atom_num"][0], 12.7279, 4)
        self.assertEqual(df_elem_deml.isna().sum().sum(), 0)

        df_elem_deml = ElementProperty.from_preset("deml", impute_nan=True).featurize_dataframe(
            self.df_nans, col_id="composition"
        )
        self.assertAlmostEqual(df_elem_deml["DemlData minimum atom_num"][0], 92)
        self.assertAlmostEqual(df_elem_deml["DemlData maximum atom_num"][0], 118)
        self.assertAlmostEqual(df_elem_deml["DemlData range atom_num"][0], 26)
        self.assertAlmostEqual(df_elem_deml["DemlData mean atom_num"][0], 107.6)
        self.assertAlmostEqual(df_elem_deml["DemlData std_dev atom_num"][0], 18.3848, 4)
        self.assertEqual(df_elem_deml.isna().sum().sum(), 0)

    def test_elem_matminer(self):
        df_elem = ElementProperty.from_preset("matminer", impute_nan=False).featurize_dataframe(
            self.df, col_id="composition"
        )
        self.assertAlmostEqual(df_elem["PymatgenData minimum melting_point"][0], 54.8, 1)
        self.assertTrue(math.isnan(df_elem["PymatgenData maximum bulk_modulus"][0]))
        self.assertAlmostEqual(df_elem["PymatgenData range X"][0], 1.61, 1)
        self.assertAlmostEqual(df_elem["PymatgenData mean X"][0], 2.796, 1)
        self.assertAlmostEqual(df_elem["PymatgenData maximum block"][0], 3, 1)
        self.assertEqual(df_elem.isna().sum().sum(), 30)

        df_elem = ElementProperty.from_preset("matminer", impute_nan=False).featurize_dataframe(
            self.df_nans, col_id="composition"
        )
        self.assertEqual(df_elem.isna().sum().sum(), 45)
        self.assertAlmostEqual(df_elem.drop(columns="composition").sum().sum(), 987.903, 3)

        df_elem = ElementProperty.from_preset("matminer", impute_nan=True).featurize_dataframe(
            self.df, col_id="composition"
        )
        self.assertAlmostEqual(df_elem["PymatgenData minimum melting_point"][0], 54.8, 1)
        self.assertFalse(math.isnan(df_elem["PymatgenData maximum bulk_modulus"][0]))
        self.assertAlmostEqual(df_elem["PymatgenData range X"][0], 1.61, 1)
        self.assertAlmostEqual(df_elem["PymatgenData mean X"][0], 2.796, 1)
        self.assertAlmostEqual(df_elem["PymatgenData maximum block"][0], 3, 1)
        self.assertEqual(df_elem.isna().sum().sum(), 0)

        df_elem = ElementProperty.from_preset("matminer", impute_nan=True).featurize_dataframe(
            self.df_nans, col_id="composition"
        )
        self.assertEqual(df_elem.isna().sum().sum(), 0)
        self.assertAlmostEqual(
            df_elem[
                [col for col in df_elem.columns if "composition" not in col and "electrical_resistivity" not in col]
            ]
            .sum()
            .sum(),
            16295.221,
            3,
        )

    def test_elem_matscholar_el(self):
        df_elem = ElementProperty.from_preset("matscholar_el", impute_nan=False).featurize_dataframe(
            self.df, col_id="composition"
        )
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

        df_elem = ElementProperty.from_preset("matscholar_el", impute_nan=False).featurize_dataframe(
            self.df_nans, col_id="composition", ignore_errors=True
        )
        self.assertEqual(df_elem.isna().sum().sum(), 1000)

        df_elem = ElementProperty.from_preset("matscholar_el", impute_nan=True).featurize_dataframe(
            self.df, col_id="composition"
        )
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
        self.assertEqual(df_elem.isna().sum().sum(), 0)

        df_elem = ElementProperty.from_preset("matscholar_el", impute_nan=True).featurize_dataframe(
            self.df_nans, col_id="composition"
        )
        self.assertEqual(df_elem.isna().sum().sum(), 0)
        self.assertAlmostEqual(df_elem.drop(columns="composition").sum().sum(), 17.666, 3)

    def test_elem_megnet_el(self):
        ep = ElementProperty.from_preset("megnet_el", impute_nan=False)
        df_elem = ep.featurize_dataframe(self.df, col_id="composition")
        self.assertAlmostEqual(df_elem["MEGNetElementData maximum embedding 1"].iloc[0], 0.127333, places=6)
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

        df_elem = ep.featurize_dataframe(self.df_nans, col_id="composition")
        self.assertAlmostEqual(df_elem["MEGNetElementData maximum embedding 1"].iloc[0], -0.044911, places=6)
        self.assertAlmostEqual(
            df_elem["MEGNetElementData maximum embedding 11"].iloc[0],
            0.191229,
            places=6,
        )
        self.assertTrue(ep.citations())

        ep = ElementProperty.from_preset("megnet_el", impute_nan=True)
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
        self.assertEqual(df_elem.isna().sum().sum(), 0)

        df_elem = ep.featurize_dataframe(self.df_nans, col_id="composition")
        self.assertAlmostEqual(df_elem["MEGNetElementData maximum embedding 1"].iloc[0], -0.001072, places=6)
        self.assertAlmostEqual(
            df_elem["MEGNetElementData maximum embedding 11"].iloc[0],
            0.191229,
            places=6,
        )
        self.assertTrue(ep.citations())
        self.assertEqual(df_elem.isna().sum().sum(), 0)

    def test_meredig(self):
        df_val = Meredig(impute_nan=False).featurize_dataframe(self.df, col_id="composition")
        self.assertAlmostEqual(df_val["Fe fraction"].iloc[0], 2.0 / 5.0)
        self.assertAlmostEqual(df_val["Fe fraction"].iloc[1], 0.5)
        self.assertAlmostEqual(df_val["O fraction"].iloc[0], 3.0 / 5.0)
        self.assertAlmostEqual(df_val["O fraction"].iloc[1], 0.5)
        self.assertAlmostEqual(df_val["frac s valence electrons"].iloc[0], 0.294117647)
        self.assertAlmostEqual(df_val["mean Number"].iloc[0], 15.2)

        df_val = Meredig(impute_nan=False).featurize_dataframe(self.df_nans, col_id="composition")
        self.assertEqual(df_val.isna().sum().sum(), 17)
        self.assertAlmostEqual(df_val.drop(columns="composition").sum().sum(), 1, 10)

        df_val = Meredig(impute_nan=True).featurize_dataframe(self.df, col_id="composition")
        self.assertAlmostEqual(df_val["Fe fraction"].iloc[0], 2.0 / 5.0)
        self.assertAlmostEqual(df_val["Fe fraction"].iloc[1], 0.5)
        self.assertAlmostEqual(df_val["O fraction"].iloc[0], 3.0 / 5.0)
        self.assertAlmostEqual(df_val["O fraction"].iloc[1], 0.5)
        self.assertAlmostEqual(df_val["frac s valence electrons"].iloc[0], 0.294117647)
        self.assertAlmostEqual(df_val["mean Number"].iloc[0], 15.2)
        self.assertEqual(df_val.isna().sum().sum(), 0)

        df_val = Meredig(impute_nan=True).featurize_dataframe(self.df_nans, col_id="composition")
        self.assertEqual(df_val.isna().sum().sum(), 0)
        self.assertAlmostEqual(df_val.drop(columns="composition").sum().sum(), 311.5897, 4)

    def test_fere_corr(self):
        df_fere_corr = ElementProperty(
            features=["FERE correction"],
            stats=["minimum", "maximum", "range", "mean", "std_dev"],
            data_source="deml",
            impute_nan=False,
        ).featurize_dataframe(self.df, col_id="composition")
        self.assertAlmostEqual(df_fere_corr["DemlData minimum FERE correction"][0], -0.15213431610903)
        self.assertAlmostEqual(df_fere_corr["DemlData maximum FERE correction"][0], 0.23)
        self.assertAlmostEqual(df_fere_corr["DemlData range FERE correction"][0], 0.382134316)
        self.assertAlmostEqual(df_fere_corr["DemlData mean FERE correction"][0], 0.077146274)
        self.assertAlmostEqual(df_fere_corr["DemlData std_dev FERE correction"][0], 0.270209766)

        df_fere_corr = ElementProperty(
            features=["FERE correction"],
            stats=["minimum", "maximum", "range", "mean", "std_dev"],
            data_source="deml",
            impute_nan=False,
        ).featurize_dataframe(self.df_nans, col_id="composition")
        self.assertEqual(df_fere_corr.isna().sum().sum(), 5)

        df_fere_corr = ElementProperty(
            features=["FERE correction"],
            stats=["minimum", "maximum", "range", "mean", "std_dev"],
            data_source="deml",
            impute_nan=True,
        ).featurize_dataframe(self.df, col_id="composition")
        self.assertAlmostEqual(df_fere_corr["DemlData minimum FERE correction"][0], -0.15213431610903)
        self.assertAlmostEqual(df_fere_corr["DemlData maximum FERE correction"][0], 0.23)
        self.assertAlmostEqual(df_fere_corr["DemlData range FERE correction"][0], 0.382134316)
        self.assertAlmostEqual(df_fere_corr["DemlData mean FERE correction"][0], 0.077146274)
        self.assertAlmostEqual(df_fere_corr["DemlData std_dev FERE correction"][0], 0.270209766)
        self.assertEqual(df_fere_corr.isna().sum().sum(), 0)

        df_fere_corr = ElementProperty(
            features=["FERE correction"],
            stats=["minimum", "maximum", "range", "mean", "std_dev"],
            data_source="deml",
            impute_nan=True,
        ).featurize_dataframe(self.df_nans, col_id="composition")
        self.assertEqual(df_fere_corr.isna().sum().sum(), 0)
        self.assertAlmostEqual(df_fere_corr.drop(columns="composition").sum().sum(), 0.2795, 4)

    def test_elem_optical(self):
        df_elem = ElementProperty.from_preset("optical", impute_nan=False).featurize_dataframe(
            self.df, col_id="composition"
        )
        self.assertAlmostEqual(df_elem["OpticalData mean n_400.0"].iloc[0], 1.98229162203492)
        self.assertAlmostEqual(df_elem["OpticalData range k_760.0"].iloc[1], 4.88738594404032)
        self.assertAlmostEqual(df_elem["OpticalData maximum R_720.0"].iloc[0], 0.621705031591809)

        df_elem = ElementProperty.from_preset("optical", impute_nan=False).featurize_dataframe(
            self.df_nans, col_id="composition"
        )
        self.assertEqual(df_elem.isna().sum().sum(), 180)

        df_elem = ElementProperty.from_preset("optical", impute_nan=True).featurize_dataframe(
            self.df, col_id="composition"
        )
        self.assertAlmostEqual(df_elem["OpticalData mean n_400.0"].iloc[0], 1.98229162203492)
        self.assertAlmostEqual(df_elem["OpticalData range k_760.0"].iloc[1], 4.88738594404032)
        self.assertAlmostEqual(df_elem["OpticalData maximum R_720.0"].iloc[0], 0.621705031591809)

        df_elem = ElementProperty.from_preset("optical", impute_nan=True).featurize_dataframe(
            self.df_nans, col_id="composition"
        )
        self.assertEqual(df_elem.isna().sum().sum(), 0)
        self.assertAlmostEqual(df_elem.drop(columns="composition").sum().sum(), 204.4712, 4)

    def test_elem_transport(self):
        df_elem = ElementProperty.from_preset("mp_transport", impute_nan=False).featurize_dataframe(
            self.df, col_id="composition"
        )
        self.assertAlmostEqual(df_elem["TransportData mean sigma_p"].iloc[0], 14933.7481377614, places=6)
        self.assertAlmostEqual(df_elem["TransportData std_dev S_n"].iloc[1], 489.973884028426)
        self.assertAlmostEqual(df_elem["TransportData mean m_p"].iloc[0], -0.00019543531213698)

        df_elem = ElementProperty.from_preset("mp_transport", impute_nan=False).featurize_dataframe(
            self.df_nans, col_id="composition"
        )
        self.assertEqual(df_elem.isna().sum().sum(), 60)

        df_elem = ElementProperty.from_preset("mp_transport", impute_nan=True).featurize_dataframe(
            self.df, col_id="composition"
        )
        self.assertAlmostEqual(df_elem["TransportData mean sigma_p"].iloc[0], 14933.7481377614, places=6)
        self.assertAlmostEqual(df_elem["TransportData std_dev S_n"].iloc[1], 489.973884028426)
        self.assertAlmostEqual(df_elem["TransportData mean m_p"].iloc[0], -0.00019543531213698)

        df_elem = ElementProperty.from_preset("mp_transport", impute_nan=True).featurize_dataframe(
            self.df_nans, col_id="composition"
        )
        self.assertAlmostEqual(df_elem.drop(columns="composition").sum().sum(), 10029874.1567, 4)


if __name__ == "__main__":
    unittest.main()
