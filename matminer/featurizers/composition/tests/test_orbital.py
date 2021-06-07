import unittest

from pymatgen.core import Composition

from matminer.featurizers.composition.orbital import (
    ValenceOrbital,
    AtomicOrbitals,
)
from matminer.featurizers.composition.tests.base import CompositionFeaturesTest


class OrbitalFeaturesTest(CompositionFeaturesTest):
    def test_valence(self):
        df_val = ValenceOrbital().featurize_dataframe(self.df, col_id="composition")
        self.assertAlmostEqual(df_val["avg s valence electrons"][0], 2.0)
        self.assertAlmostEqual(df_val["avg p valence electrons"][0], 2.4)
        self.assertAlmostEqual(df_val["avg d valence electrons"][0], 2.4)
        self.assertAlmostEqual(df_val["avg f valence electrons"][0], 0.0)
        self.assertAlmostEqual(df_val["frac s valence electrons"][0], 0.294117647)
        self.assertAlmostEqual(df_val["frac d valence electrons"][0], 0.352941176)
        self.assertAlmostEqual(df_val["frac p valence electrons"][0], 0.352941176)
        self.assertAlmostEqual(df_val["frac f valence electrons"][0], 0)

    def test_atomic_orbitals(self):
        df_atomic_orbitals = AtomicOrbitals().featurize_dataframe(self.df, col_id="composition")
        self.assertEqual(df_atomic_orbitals["HOMO_character"][0], "d")
        self.assertEqual(df_atomic_orbitals["HOMO_element"][0], "Fe")
        self.assertEqual(df_atomic_orbitals["HOMO_energy"][0], -0.295049)
        self.assertEqual(df_atomic_orbitals["LUMO_character"][0], "d")
        self.assertEqual(df_atomic_orbitals["LUMO_element"][0], "Fe")
        self.assertEqual(df_atomic_orbitals["LUMO_energy"][0], -0.295049)
        self.assertEqual(df_atomic_orbitals["gap_AO"][0], 0.0)

        # test that fractional compositions return the same features
        self.assertEqual(
            AtomicOrbitals().featurize(Composition("Na0.5Cl0.5")),
            AtomicOrbitals().featurize(Composition("NaCl")),
        )

        # test if warning is raised upon composition truncation in dilute cases
        self.assertWarns(UserWarning, AtomicOrbitals().featurize, Composition("Fe1C0.00000001"))


if __name__ == "__main__":
    unittest.main()
