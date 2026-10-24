import unittest
from types import SimpleNamespace
from unittest import SkipTest

from pymatgen.core import Composition

from matminer.featurizers.composition.thermo import CohesiveEnergy, CohesiveEnergyMP

from ...mp_api_mock import FakeMPRester, fake_mp_api
from .conftest import CompositionFeaturesTest


def _get_mprester():
    try:
        from mp_api.client import MPRester
        from mp_api.client.core import MPRestError
    except ImportError:
        raise SkipTest("mp_api not installed; Skipping cohesive energy test")
    try:
        mpr = MPRester()
    except (ValueError, MPRestError):
        raise SkipTest("Materials Project API key not set; Skipping cohesive energy test")
    if not mpr.api_key:
        raise SkipTest("Materials Project API key not set; Skipping cohesive energy test")
    return mpr


class _MPRester(FakeMPRester):
    docs = [
        SimpleNamespace(material_id="mp-2", energy_per_atom=-6.0, formation_energy_per_atom=-1.0),
        SimpleNamespace(material_id="mp-1", energy_per_atom=-7.0, formation_energy_per_atom=-2.0),
    ]
    cohesive_energy_args = None

    def get_cohesive_energy(self, material_ids, normalization="atom"):
        type(self).cohesive_energy_args = (material_ids, normalization)
        return {mpid: 5.5 for mpid in material_ids}


class ThermoFeaturesTest(CompositionFeaturesTest):
    def test_cohesive_energy(self):
        _get_mprester()

        df_cohesive_energy = CohesiveEnergy().featurize_dataframe(self.df, col_id="composition")
        self.assertAlmostEqual(df_cohesive_energy["cohesive energy"][0], 5.179358342, 2)

    def test_cohesive_energy_mp(self):
        raise SkipTest("Unable to debug issues with this test without a legacy MP key. Skipping for now.")
        _get_mprester()
        ce = CohesiveEnergyMP()
        ce.set_n_jobs(1)
        df_cohesive_energy = ce.featurize_dataframe(self.df, col_id="composition")
        self.assertAlmostEqual(df_cohesive_energy["cohesive energy (MP)"][0], 5.778053364, 2)

    def test_cohesive_energy_uses_mp_api(self):
        comp = Composition("Fe2O3")
        with fake_mp_api(_MPRester):
            cohesive_energy = CohesiveEnergy(mapi_key="key").featurize(comp)
        self.assertEqual(_MPRester.search_kwargs["formula"], "Fe2O3")
        # The most stable entry (lowest energy_per_atom) is mp-1
        self.assertEqual(cohesive_energy, CohesiveEnergy().featurize(comp, formation_energy_per_atom=-2.0))

    def test_cohesive_energy_mp_uses_mp_api(self):
        with fake_mp_api(_MPRester):
            cohesive_energy = CohesiveEnergyMP(mapi_key="key").featurize(Composition("Fe2O3"))
        self.assertEqual(_MPRester.search_kwargs["formula"], "Fe2O3")
        self.assertEqual(_MPRester.cohesive_energy_args, (["mp-1"], "atom"))
        self.assertEqual(cohesive_energy, [5.5])


if __name__ == "__main__":
    unittest.main()
