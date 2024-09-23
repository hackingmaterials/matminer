import unittest
from unittest import SkipTest

from pymatgen.ext.matproj import MPRester, MPRestError

from matminer.featurizers.composition.tests.base import CompositionFeaturesTest
from matminer.featurizers.composition.thermo import CohesiveEnergy, CohesiveEnergyMP


class ThermoFeaturesTest(CompositionFeaturesTest):
    def test_cohesive_energy(self):
        try:
            mpr = MPRester()
        except (ValueError, MPRestError):
            raise SkipTest("Materials Project API key not set; Skipping cohesive energy test")

        if not mpr.api_key:
            raise SkipTest("Materials Project API key not set; Skipping cohesive energy test")

        df_cohesive_energy = CohesiveEnergy().featurize_dataframe(self.df, col_id="composition")
        self.assertAlmostEqual(df_cohesive_energy["cohesive energy"][0], 5.179358342, 2)

    def test_cohesive_energy_mp(self):
        raise SkipTest("Unable to debug issues with this test without a legacy MP key. Skipping for now.")
        try:
            mpr = MPRester()
        except (ValueError, MPRestError):
            raise SkipTest("Materials Project API key not set; Skipping cohesive energy test")

        if not mpr.api_key:
            raise SkipTest("Materials Project API key not set; Skipping cohesive energy test")
        ce = CohesiveEnergyMP()
        ce.set_n_jobs(1)
        df_cohesive_energy = ce.featurize_dataframe(self.df, col_id="composition")
        self.assertAlmostEqual(df_cohesive_energy["cohesive energy (MP)"][0], 5.778053364, 2)


if __name__ == "__main__":
    unittest.main()
