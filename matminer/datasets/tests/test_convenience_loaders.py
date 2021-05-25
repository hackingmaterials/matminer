import pandas as pd
import unittest

from matminer.datasets.tests.base import DatasetTest, do_complete_test
from matminer.datasets.convenience_loaders import (
    load_glass_ternary_hipt,
    load_castelli_perovskites,
    load_flla,
    load_boltztrap_mp,
    load_citrine_thermal_conductivity,
    load_dielectric_constant,
    load_double_perovskites_gap,
    load_double_perovskites_gap_lumo,
    load_elastic_tensor,
    load_glass_ternary_landolt,
    load_phonon_dielectric_mp,
    load_piezoelectric_tensor,
    load_mp,
    load_wolverton_oxides,
    load_heusler_magnetic,
    load_steel_strength,
    load_jarvis_ml_dft_training,
    load_jarvis_dft_2d,
    load_jarvis_dft_3d,
    load_expt_formation_enthalpy,
    load_expt_gap,
    load_glass_binary,
    load_m2ax,
    load_brgoch_superhard_training,
)

if do_complete_test:

    class ConvenienceLoadersTest(DatasetTest):
        def test_load_glass_ternary_hipt(self):
            df = load_glass_ternary_hipt()
            self.assertTrue(isinstance(df, pd.DataFrame))

            with self.assertRaises(AttributeError):
                df = load_glass_ternary_hipt(system="Nonexistent system")

            df = load_glass_ternary_hipt(system="CoFeZr")
            self.assertEqual(len(df.index), 1295)

            df = load_glass_ternary_hipt(system=["CoFeZr", "CoTiZr"])
            self.assertEqual(len(df.index), 2576)

        def test_load_castelli_peroskites(self):
            df = load_castelli_perovskites()
            self.assertTrue(isinstance(df, pd.DataFrame))

        def test_load_flla(self):
            df = load_flla()
            self.assertTrue(isinstance(df, pd.DataFrame))

        def test_load_boltztrap_mp(self):
            df = load_boltztrap_mp()
            self.assertTrue(isinstance(df, pd.DataFrame))

        def test_load_citrine_thermal_conductivity(self):
            df = load_citrine_thermal_conductivity(room_temperature=False)
            self.assertTrue(isinstance(df, pd.DataFrame))
            self.assertEqual(len(df), 872)

            df = load_citrine_thermal_conductivity(room_temperature=True)
            self.assertTrue(isinstance(df, pd.DataFrame))
            self.assertEqual(len(df), 295)

        def test_load_double_perovskites_gap(self):
            df = load_double_perovskites_gap()
            self.assertTrue(isinstance(df, pd.DataFrame))

            df = load_double_perovskites_gap(return_lumo=True)
            self.assertTrue(isinstance(df, tuple))

        def test_load_double_perovskites_gap_lumo(self):
            df = load_double_perovskites_gap_lumo()
            self.assertTrue(isinstance(df, pd.DataFrame))

        def test_load_dielectric_constant(self):
            df = load_dielectric_constant()
            self.assertTrue(isinstance(df, pd.DataFrame))
            self.assertEqual(len(df.columns), 13)

            df = load_dielectric_constant(include_metadata=True)
            self.assertTrue(isinstance(df, pd.DataFrame))
            self.assertEqual(len(df.columns), 16)

        def test_load_piezoelectric_tensor(self):
            df = load_piezoelectric_tensor()
            self.assertTrue(isinstance(df, pd.DataFrame))
            self.assertEqual(len(df.columns), 10)

            df = load_piezoelectric_tensor(include_metadata=True)
            self.assertTrue(isinstance(df, pd.DataFrame))
            self.assertEqual(len(df.columns), 13)

        def test_load_elastic_tensor(self):
            df = load_elastic_tensor()
            self.assertTrue(isinstance(df, pd.DataFrame))
            self.assertEqual(len(df.columns), 17)

            df = load_elastic_tensor(include_metadata=True)
            self.assertTrue(isinstance(df, pd.DataFrame))
            self.assertEqual(len(df.columns), 20)

        def test_load_glass_ternary_landolt(self):
            df = load_glass_ternary_landolt()
            self.assertTrue(isinstance(df, pd.DataFrame))
            self.assertEqual(len(df), 6118)

            df = load_glass_ternary_landolt(unique_composition=False)
            self.assertTrue(isinstance(df, pd.DataFrame))
            self.assertEqual(len(df), 7191)

            df = load_glass_ternary_landolt(processing="meltspin")
            self.assertTrue(isinstance(df, pd.DataFrame))
            self.assertEqual(len(df), 5736)

            df = load_glass_ternary_landolt(processing="sputtering")
            self.assertTrue(isinstance(df, pd.DataFrame))
            self.assertEqual(len(df), 403)

            with self.assertRaises(ValueError):
                df = load_glass_ternary_landolt(processing="spittering")

        def test_load_phonon_dielectric_mp(self):
            df = load_phonon_dielectric_mp()
            self.assertTrue(isinstance(df, pd.DataFrame))

        def test_load_glass_binary(self):
            df = load_glass_binary()
            self.assertTrue(isinstance(df, pd.DataFrame))
            self.assertEqual(len(df), 5483)
            df = load_glass_binary(version="v1")
            self.assertTrue(isinstance(df, pd.DataFrame))
            self.assertEqual(len(df), 5959)

        def test_load_m2ax(self):
            df = load_m2ax()
            self.assertTrue(isinstance(df, pd.DataFrame))

        def test_load_expt_gap(self):
            df = load_expt_gap()
            self.assertTrue(isinstance(df, pd.DataFrame))

        def test_load_expt_formation_enthalpy(self):
            df = load_expt_formation_enthalpy()
            self.assertTrue(isinstance(df, pd.DataFrame))

        def test_load_mp(self):
            df = load_mp()
            self.assertTrue(isinstance(df, pd.DataFrame))
            self.assertEqual(len(df.columns), 9)

            df = load_mp(include_structures=True)
            self.assertTrue(isinstance(df, pd.DataFrame))
            self.assertEqual(len(df.columns), 11)

        def test_load_wolverton_oxides(self):
            df = load_wolverton_oxides()
            self.assertTrue(isinstance(df, pd.DataFrame))

        def test_load_heusler_magnetic(self):
            df = load_heusler_magnetic()
            self.assertTrue(isinstance(df, pd.DataFrame))

        def test_load_steel_strength(self):
            df = load_steel_strength()
            self.assertTrue(isinstance(df, pd.DataFrame))

        def test_load_jarvis_ml_dft_training(self):
            df = load_jarvis_ml_dft_training()
            self.assertTrue(isinstance(df, pd.DataFrame))
            self.assertEqual(len(df), 24759)
            df = load_jarvis_ml_dft_training(drop_nan_columns="e mass_x")
            self.assertEqual(len(df), 19978)

        def test_load_jarvis_dft_2d(self):
            df = load_jarvis_dft_2d()
            self.assertTrue(isinstance(df, pd.DataFrame))
            self.assertEqual(len(df), 636)
            df = load_jarvis_dft_2d(drop_nan_columns="epsilon_x opt")
            self.assertEqual(len(df), 522)

        def test_load_jarvis_dft_3d(self):
            df = load_jarvis_dft_3d()
            self.assertTrue(isinstance(df, pd.DataFrame))
            self.assertEqual(len(df), 25923)
            df = load_jarvis_dft_3d(drop_nan_columns="epsilon_x opt")
            self.assertEqual(len(df), 19027)

        def test_load_brgoch_superhard_training(self):
            df = load_brgoch_superhard_training()
            self.assertTrue(isinstance(df, pd.DataFrame))
            self.assertEqual(len(df.columns), 158)
            self.assertEqual(len(df), 2574)
            df = load_brgoch_superhard_training(drop_suspect=True)
            self.assertEqual(len(df), 2494)
            df = load_brgoch_superhard_training(subset="brgoch_features")
            self.assertEqual(len(df), 2574)
            self.assertEqual(len(df.columns), 152)
            df = load_brgoch_superhard_training(subset="basic_descriptors")
            self.assertEqual(len(df), 2574)
            self.assertEqual(len(df.columns), 4)
            df = load_brgoch_superhard_training(subset="basic_descriptors", drop_suspect=True)
            self.assertEqual(len(df), 2494)
            self.assertEqual(len(df.columns), 4)


if __name__ == "__main__":
    unittest.main()
