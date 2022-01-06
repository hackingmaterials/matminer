import copy
import unittest

import numpy as np
import pandas as pd
from pymatgen.core import Lattice, Structure
from sklearn.exceptions import NotFittedError

from matminer.featurizers.structure.bonding import (
    BagofBonds,
    BondFractions,
    GlobalInstabilityIndex,
    MinimumRelativeDistances,
    StructuralHeterogeneity,
)
from matminer.featurizers.structure.matrix import CoulombMatrix, SineCoulombMatrix
from matminer.featurizers.structure.tests.base import StructureFeaturesTest


class BondingStructureTest(StructureFeaturesTest):
    def test_bondfractions(self):

        # Test individual structures with featurize
        bf_md = BondFractions.from_preset("MinimumDistanceNN")
        bf_md.no_oxi = True
        bf_md.fit([self.diamond_no_oxi])
        self.assertArrayEqual(bf_md.featurize(self.diamond), [1.0])
        self.assertArrayEqual(bf_md.featurize(self.diamond_no_oxi), [1.0])

        bf_voronoi = BondFractions.from_preset("VoronoiNN")
        bf_voronoi.bbv = float("nan")
        bf_voronoi.fit([self.nacl])
        bond_fracs = bf_voronoi.featurize(self.nacl)
        bond_names = bf_voronoi.feature_labels()
        ref = {
            "Na+ - Na+ bond frac.": 0.25,
            "Cl- - Na+ bond frac.": 0.5,
            "Cl- - Cl- bond frac.": 0.25,
        }
        self.assertDictEqual(dict(zip(bond_names, bond_fracs)), ref)

        # Test to make sure dataframe behavior is as intended
        s_list = [self.diamond_no_oxi, self.ni3al]
        df = pd.DataFrame.from_dict({"s": s_list})
        bf_voronoi.fit(df["s"])
        df = bf_voronoi.featurize_dataframe(df, "s")

        # Ensure all data is properly labelled and organized
        self.assertArrayEqual(df["C - C bond frac."].to_numpy(), [1.0, np.nan])
        self.assertArrayEqual(df["Al - Ni bond frac."].to_numpy(), [np.nan, 0.5])
        self.assertArrayEqual(df["Al - Al bond frac."].to_numpy(), [np.nan, 0.0])
        self.assertArrayEqual(df["Ni - Ni bond frac."].to_numpy(), [np.nan, 0.5])

        # Test to make sure bad_bond_values (bbv) are still changed correctly
        # and check inplace behavior of featurize dataframe.
        bf_voronoi.bbv = 0.0
        df = pd.DataFrame.from_dict({"s": s_list})
        df = bf_voronoi.featurize_dataframe(df, "s")
        self.assertArrayEqual(df["C - C bond frac."].to_numpy(), [1.0, 0.0])
        self.assertArrayEqual(df["Al - Ni bond frac."].to_numpy(), [0.0, 0.5])
        self.assertArrayEqual(df["Al - Al bond frac."].to_numpy(), [0.0, 0.0])
        self.assertArrayEqual(df["Ni - Ni bond frac."].to_numpy(), [0.0, 0.5])

    def test_bob(self):

        # Test a single fit and featurization
        scm = SineCoulombMatrix(flatten=False)
        bob = BagofBonds(coulomb_matrix=scm, token=" - ")
        bob.fit([self.ni3al])
        truth1 = [
            235.74041833262768,
            1486.4464890775491,
            1486.4464890775491,
            1486.4464890775491,
            38.69353092306119,
            38.69353092306119,
            38.69353092306119,
            38.69353092306119,
            38.69353092306119,
            38.69353092306119,
            83.33991275736257,
            83.33991275736257,
            83.33991275736257,
            83.33991275736257,
            83.33991275736257,
            83.33991275736257,
        ]
        truth1_labels = [
            "Al site #0",
            "Ni site #0",
            "Ni site #1",
            "Ni site #2",
            "Al - Ni bond #0",
            "Al - Ni bond #1",
            "Al - Ni bond #2",
            "Al - Ni bond #3",
            "Al - Ni bond #4",
            "Al - Ni bond #5",
            "Ni - Ni bond #0",
            "Ni - Ni bond #1",
            "Ni - Ni bond #2",
            "Ni - Ni bond #3",
            "Ni - Ni bond #4",
            "Ni - Ni bond #5",
        ]
        self.assertArrayAlmostEqual(bob.featurize(self.ni3al), truth1)
        self.assertEqual(bob.feature_labels(), truth1_labels)

        # Test padding from fitting and dataframe featurization
        bob.coulomb_matrix = CoulombMatrix(flatten=False)
        bob.fit([self.ni3al, self.cscl, self.diamond_no_oxi])
        df = pd.DataFrame({"structures": [self.cscl]})
        df = bob.featurize_dataframe(df, "structures")
        self.assertEqual(len(df.columns.values), 25)
        self.assertAlmostEqual(df["Cs+ site #0"][0], 7513.468312122532)
        self.assertAlmostEqual(df["Al site #0"][0], 0.0)
        self.assertAlmostEqual(df["Cs+ - Cl- bond #1"][0], 135.74726437398044, 3)
        self.assertAlmostEqual(df["Al - Ni bond #0"][0], 0.0)

        # Test error handling for bad fits or null fits
        bob = BagofBonds(CoulombMatrix(flatten=False))
        self.assertRaises(NotFittedError, bob.featurize, self.nacl)
        bob.fit([self.ni3al, self.diamond])
        self.assertRaises(ValueError, bob.featurize, self.nacl)

    def test_ward_prb_2017_strhet(self):
        f = StructuralHeterogeneity()

        # Test Ni3Al, which is uniform
        features = f.featurize(self.ni3al)
        self.assertArrayAlmostEqual([0, 1, 1, 0, 0, 0, 0, 0, 0], features)

        # Do CsCl, which has variation in the neighbors
        big_face_area = np.sqrt(3) * 3 / 2 * (2 / 4 / 4)
        small_face_area = 0.125
        average_dist = (8 * np.sqrt(3) / 2 * big_face_area + 6 * small_face_area) / (
            8 * big_face_area + 6 * small_face_area
        )
        rel_var = (
            (8 * abs(np.sqrt(3) / 2 - average_dist) * big_face_area + 6 * abs(1 - average_dist) * small_face_area)
            / (8 * big_face_area + 6 * small_face_area)
            / average_dist
        )
        cscl = Structure(
            Lattice([[4.209, 0, 0], [0, 4.209, 0], [0, 0, 4.209]]),
            ["Cl1-", "Cs1+"],
            [[0.5, 0.5, 0.5], [0, 0, 0]],
            validate_proximity=False,
            to_unit_cell=False,
            coords_are_cartesian=False,
            site_properties=None,
        )
        features = f.featurize(cscl)
        self.assertArrayAlmostEqual([0, 1, 1, rel_var, rel_var, 0, rel_var, 0, 0], features)

    def test_GlobalInstabilityIndex(self):
        # Test diamond and ni3al fail precheck
        gii = GlobalInstabilityIndex(r_cut=4.0, disordered_pymatgen=False)
        self.assertFalse(gii.precheck(self.diamond))
        self.assertFalse(gii.precheck(self.ni3al))
        # Test they raise errors when featurizing
        with self.assertRaises(AttributeError):
            gii.featurize(self.ni3al)
        with self.assertRaises(ValueError):
            gii.featurize(self.diamond)

        # Ordinary case of nacl
        self.assertTrue(gii.precheck(self.nacl))
        self.assertAlmostEqual(gii.featurize(self.nacl)[0], 0.08491655709)

        # Test bond valence sums are accurate for NaCl.
        # Values are closer to 0.915 than 1.0 due to structure specified here.
        # Using CollCode181148 from the ICSD, I see bond valence sums of 0.979
        site1, site2 = (self.nacl[0], self.nacl[1])
        neighs1 = self.nacl.get_neighbors(site1, 4)
        neighs2 = self.nacl.get_neighbors(site2, 4)
        site_val1 = site1.species.elements[0].oxi_state
        site_el1 = str(site1.species.element_composition.elements[0])
        site_val2 = site2.species.elements[0].oxi_state
        site_el2 = str(site2.species.element_composition.elements[0])
        self.assertAlmostEqual(gii.calc_bv_sum(site_val1, site_el1, neighs1), 0.9150834429025214)
        self.assertAlmostEqual(gii.calc_bv_sum(site_val2, site_el2, neighs2), -0.915083442902522)

        # Behavior when disorder is present
        gii_pymat = GlobalInstabilityIndex(r_cut=4.0, disordered_pymatgen=True)
        nacl_disordered = copy.deepcopy(self.nacl)
        nacl_disordered.replace_species({"Cl1-": "Cl0.5Br0.5"})
        nacl_disordered.add_oxidation_state_by_element({"Na": 1, "Cl": -1, "Br": -1})
        self.assertTrue(gii.precheck(nacl_disordered))
        with self.assertRaises(ValueError):
            gii.featurize(nacl_disordered)
        self.assertAlmostEqual(gii_pymat.featurize(nacl_disordered)[0], 0.39766464)

    def test_min_relative_distances(self):
        with self.assertRaises(ValueError):
            MinimumRelativeDistances(include_species=False, include_distances=False)

        mrd_nonuniform = MinimumRelativeDistances(flatten=False)
        self.assertAlmostEqual(mrd_nonuniform.featurize(self.diamond_no_oxi)[0][0], 1.1052576)
        self.assertAlmostEqual(mrd_nonuniform.featurize(self.nacl)[0][0], 0.8891443)
        self.assertAlmostEqual(mrd_nonuniform.featurize(self.cscl)[0][0], 0.9877540)

        mrd_flat = MinimumRelativeDistances(flatten=True)

        with self.assertRaises(NotFittedError):
            mrd_flat.featurize(self.diamond)

        # Fit on a structure with 2 sites:
        mrd_flat.fit([self.diamond_no_oxi])

        # Ensure it can featurize the structure it was fit on
        f_diamond = mrd_flat.featurize(self.diamond_no_oxi)
        self.assertAlmostEqual(f_diamond[0], 1.1052576)
        self.assertEqual(f_diamond[1], "C")
        self.assertEqual(f_diamond[2], "C")
        self.assertAlmostEqual(f_diamond[3], 1.1052576)
        self.assertEqual(f_diamond[4], "C")
        self.assertEqual(f_diamond[5], "C")
        self.assertEqual(len(f_diamond), 6)

        # Ensure it can featurize a different structure w/ same n_sites (2)
        f_cscl = mrd_flat.featurize(self.cscl)
        self.assertAlmostEqual(f_cscl[0], 0.9877540)
        self.assertEqual(f_cscl[1], "Cl-")
        self.assertEqual(f_cscl[2][0], "Cl-")
        self.assertEqual(len(f_cscl[2]), 4)
        self.assertEqual(len(f_cscl), 6)

        # Ensure it truncates extra sites on structure w/ more n_sites
        f_ni3al = mrd_flat.featurize(self.ni3al)
        self.assertAlmostEqual(f_ni3al[0], 0.95731379)
        self.assertEqual(f_ni3al[1], "Al")
        self.assertEqual(f_ni3al[2][0], "Al")
        self.assertEqual(len(f_ni3al[2]), 12)
        self.assertEqual(len(f_ni3al), 6)
        self.assertAlmostEqual(f_ni3al[3], 0.921857729)

        # Ensure it extends extra sites on structure with fewer n_sites
        f_sc = mrd_flat.featurize(self.sc)
        self.assertAlmostEqual(f_sc[0], 1.408)
        self.assertEqual(f_sc[1], "Al")
        self.assertEqual(f_sc[2][0], "Al")
        self.assertEqual(len(f_sc[2]), 6)
        self.assertEqual(len(f_sc), 6)
        self.assertTrue(f_sc[3], np.nan)


if __name__ == "__main__":
    unittest.main()
