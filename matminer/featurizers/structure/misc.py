from __future__ import division, unicode_literals, print_function

import warnings
import math
from functools import lru_cache

import numpy as np
from pymatgen import Structure
from pymatgen.analysis import bond_valence
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.analysis.local_env import ValenceIonicRadiusEvaluator
from pymatgen.core.periodic_table import Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.structure import SymmetrizedStructure

from matminer.featurizers.base import BaseFeaturizer
from matminer.utils.data import IUCrBondValenceData


class DensityFeatures(BaseFeaturizer):
    """
    Calculates density and density-like features

    Features:
        - density
        - volume per atom
        - ("vpa"), and packing fraction
    """

    def __init__(self, desired_features=None):
        """
        Args:
            desired_features: [str] - choose from "density", "vpa",
                "packing fraction"
        """
        self.features = ["density", "vpa", "packing fraction"] if not \
            desired_features else desired_features

    def precheck(self, s: Structure) -> bool:
        """
        Precheck a single entry. DensityFeatures does not work for disordered
        structures. To precheck an entire dataframe (qnd automatically gather
        the fraction of structures that will pass the precheck), please use
        precheck_dataframe.

        Args:
            s (pymatgen.Structure): The structure to precheck.

        Returns:
            (bool): If True, s passed the precheck; otherwise, it failed.
        """
        return s.is_ordered

    def featurize(self, s):
        output = []

        if "density" in self.features:
            output.append(s.density)

        if "vpa" in self.features:
            if not s.is_ordered:
                raise ValueError("Disordered structure support not built yet.")
            output.append(s.volume / len(s))

        if "packing fraction" in self.features:
            if not s.is_ordered:
                raise ValueError("Disordered structure support not built yet.")
            total_rad = 0
            for site in s:
                total_rad += site.specie.atomic_radius ** 3
            output.append(4 * math.pi * total_rad / (3 * s.volume))

        return output

    def feature_labels(self):
        all_features = ["density", "vpa", "packing fraction"]  # enforce order
        return [x for x in all_features if x in self.features]

    def citations(self):
        return []

    def implementors(self):
        return ["Saurabh Bajaj", "Anubhav Jain"]


class GlobalInstabilityIndex(BaseFeaturizer):
    """
    The global instability index of a structure.

    The default is to use IUCr 2016 bond valence parameters for computing
    bond valence sums. If the structure has disordered site occupancies
    or non-integer valences on sites, pymatgen's bond valence sum method
    can be used instead.

    Note that pymatgen's bond valence sum method is prone to error unless
    the correct scale factor is supplied. A scale factor based on testing
    with perovskites is used here.
    TODO: Use scipy to optimize scale factor for minimizing GII

    Based on the following publication:

    'Structural characterization of R2BaCuO5 (R = Y, Lu, Yb, Tm, Er, Ho,
        Dy, Gd, Eu and Sm) oxides by X-ray and neutron diffraction',
        A.Salinas-Sanchez, J.L.Garcia-Muñoz, J.Rodriguez-Carvajal,
        R.Saez-Puche, and J.L.Martinez, Journal of Solid State Chemistry,
        100, 201-211 (1992),
        https://doi.org/10.1016/0022-4596(92)90094-C

    Args:
        r_cut: Float, how far to search for neighbors when computing bond valences
        disordered_pymatgen: Boolean, whether to fall back on pymatgen's bond
            valence sum method for disordered structures

    Features:
        The global instability index is the square root of the sum of squared
            differences of the bond valence sums from the formal valences
            averaged over all atoms in the unit cell.
    """

    def __init__(self, r_cut=4.0, disordered_pymatgen=False):

        bv = IUCrBondValenceData()
        self.bv_values = bv.params
        self.r_cut = r_cut
        self.disordered_pymatgen = disordered_pymatgen

    def precheck(self, struct):
        """
        Bond valence methods require atom pairs with oxidation states.

        Additionally, check if at least the first and last site's species
        have a entry in the bond valence parameters.

        Args:
            struct: Pymatgen Structure
        """

        anions = [
            "O", "N", "F", "Cl", "Br", "S", "Se", "I", "Te", "P", "H", "As"
        ]


        for site in struct:
            # Fail if site doesn't have either attribute
            if not hasattr(site, "species"):
                return False
            if isinstance(site.species.elements[0], Element):
                return False

        elems = [str(x.element) for x in struct.composition.elements]

        # If compound is not ionically bonded, it is going to fail
        if not any([e in anions for e in elems]):
            return False
        valences = [site.species.elements[0].oxi_state for site in struct]

        # If the oxidation states are technically provided but any are 0, fails
        if not all(valences):
            return False

        if len(struct) > 200:
            warnings.warn(
                "Computing bond valence sums for over 200 sites. "
                "Featurization might be very slow!"
            )

        # Check that all cation-anion pairs are tabulated
        specs = struct.composition.elements.copy()
        while len(specs) > 1:
            spec1 = specs.pop()
            elem1 = str(spec1.element)
            val_1 = spec1.oxi_state
            for spec2 in specs:
                elem2 = str(spec2.element)
                val_2 = spec2.oxi_state
                if np.sign(val_1) == -1 and  np.sign(val_2) == 1:
                    try:
                        self.get_bv_params(elem2, elem1, val_2, val_1)
                    except IndexError:
                        return False
        return True

    def featurize(self, struct):
        """
        Get global instability index.

        Args:
            struct: Pymatgen Structure object
        Returns:
            [gii]: Length 1 list with float value
        """

        if struct.is_ordered:
            gii = self.calc_gii_iucr(struct)
            if gii > 0.6:
                warnings.warn("GII extremely large. Table parameters may "
                                "not be suitable or structure may be unusual.")

        else:
            if self.disordered_pymatgen:
                gii = self.calc_gii_pymatgen(struct, scale_factor=0.965)
                if gii > 0.6:
                    warnings.warn(
                        "GII extremely large. Pymatgen method may not be "
                        "suitable or structure may be unusual."
                    )
                return [gii]
            else:
                raise ValueError(
                    'Structure must be ordered for table lookup method.'
                )

        return [gii]

    def get_equiv_sites(self, s, site):
        """Find identical sites from analyzing space group symmetry."""
        sga = SpacegroupAnalyzer(s, symprec=0.01)
        sg = sga.get_space_group_operations
        sym_data = sga.get_symmetry_dataset()
        equiv_atoms = sym_data["equivalent_atoms"]
        wyckoffs = sym_data["wyckoffs"]
        sym_struct = SymmetrizedStructure(s, sg, equiv_atoms, wyckoffs)
        equivs = sym_struct.find_equivalent_sites(site)
        return equivs

    def calc_bv_sum(self, site_val, site_el, neighbor_list):
        """Computes bond valence sum for site.
        Args:
            site_val (Integer): valence of site
            site_el (String): element name
            neighbor_list (List): List of neighboring sites and their distances
        """
        bvs = 0
        for neighbor_info in neighbor_list:
            neighbor = neighbor_info[0]
            dist = neighbor_info[1]
            neighbor_val = neighbor.species.elements[0].oxi_state
            neighbor_el = str(
                    neighbor.species.element_composition.elements[0])
            if neighbor_val % 1 != 0 or site_val % 1 != 0:
                raise ValueError('Some sites have non-integer valences.')
            try:
                if np.sign(site_val) == 1 and np.sign(neighbor_val) == -1:
                    params = self.get_bv_params(cation=site_el,
                                               anion=neighbor_el,
                                               cat_val=site_val,
                                               an_val=neighbor_val)
                    bvs += self.compute_bv(params, dist)
                elif np.sign(site_val) == -1 and np.sign(neighbor_val) == 1:
                    params = self.get_bv_params(cation=neighbor_el,
                                               anion=site_el,
                                               cat_val=neighbor_val,
                                               an_val=site_val)
                    bvs -= self.compute_bv(params, dist)
            except:
                raise ValueError(
                    'BV parameters for {} with valence {} and {} {} not '
                    'found in table'
                    ''.format(site_el,
                              site_val,
                              neighbor_el,
                              neighbor_val))
        return bvs

    def calc_gii_iucr(self, s):
        """Computes global instability index using tabulated bv params.

        Args:
            s: Pymatgen Structure object
        Returns:
            gii: Float, the global instability index
        """
        elements = [str(i) for i in s.composition.element_composition.elements]
        if elements[0] == elements[-1]:
            raise ValueError("No oxidation states with single element.")

        bond_valence_sums = []
        cutoff = self.r_cut
        pairs = s.get_all_neighbors(r=cutoff)
        site_val_sums = {} # Cache bond valence deviations

        for i, neighbor_list in enumerate(pairs):
            site = s[i]
            equivs = self.get_equiv_sites(s, site)
            flag = False

            # If symm. identical site has cached bond valence sum difference,
            # use it to avoid unnecessary calculations
            for item in equivs:
                if item in site_val_sums:
                    bond_valence_sums.append(site_val_sums[item])
                    site_val_sums[site] = site_val_sums[item]
                    flag = True
                    break
            if flag:
                continue
            site_val = site.species.elements[0].oxi_state
            site_el = str(site.species.element_composition.elements[0])
            bvs = self.calc_bv_sum(site_val, site_el, neighbor_list)

            site_val_sums[site] = bvs - site_val
        gii = np.linalg.norm(list(site_val_sums.values())) /\
              np.sqrt(len(site_val_sums))
        return gii

    # Cache bond valence parameters
    @lru_cache(maxsize=512)
    def get_bv_params(self, cation, anion, cat_val, an_val):
        """Lookup bond valence parameters from IUPAC table.
        Args:
            cation: String, cation element
            anion: String, anion element
            cat_val: Integer, cation formal valence
            an_val: Integer, anion formal valence
        Returns:
            bond_val_list: dataframe of bond valence parameters
        """
        bv_data = self.bv_values
        bond_val_list = bv_data[(bv_data['Atom1'] == cation) &
                                (bv_data['Atom1_valence'] == cat_val) &
                                (bv_data['Atom2'] == anion) &
                                (bv_data['Atom2_valence'] == an_val)]
        # If multiple values exist, take first one
        return bond_val_list.iloc[0]

    @staticmethod
    def compute_bv(params, dist):
        """Compute bond valence from parameters.
        Args:
            params: Dataframe with Ro and B parameters
            dist: Float, distance to neighboring atom
        Returns:
            bv: Float, bond valence
        """
        bv = np.exp((params['Ro'] - dist)/params['B'])
        return bv

    def calc_gii_pymatgen(self, struct, scale_factor=0.965):
        """Calculates global instability index using Pymatgen's bond valence sum
        Args:
            struct: Pymatgen Structure object
            scale: Float, tunable scale factor for bond valence
        Returns:
            gii: Float, global instability index
        """
        deviations = []
        cutoff=self.r_cut
        if struct.is_ordered:
            for site in struct:
                nn = struct.get_neighbors(site,r=cutoff)
                bvs = bond_valence.calculate_bv_sum(site,
                                                    nn,
                                                    scale_factor=scale_factor)
                deviations.append(bvs - site.species.elements[0].oxi_state)
            gii = np.linalg.norm(deviations) / np.sqrt(len(deviations))
        else:
            for site in struct:
                nn = struct.get_neighbors(site,r=cutoff)
                bvs = bond_valence.calculate_bv_sum_unordered(site,
                                                              nn,
                                                              scale_factor=scale_factor)
                min_diff = min(
                    [bvs - spec.oxi_state for spec in site.species.elements]
                )
                deviations.append(min_diff)
            gii = np.linalg.norm(deviations) / np.sqrt(len(deviations))
        return gii

    def feature_labels(self):
        return ["global instability index"]

    def implementors(self):
        return ["Nicholas Wagner", "Nenian Charles", "Alex Dunn"]

    def citations(self):
        return ["@article{PhysRevB.87.184115,"
                "title = {Structural characterization of R2BaCuO5 (R = Y, Lu, Yb, Tm, Er, Ho,"
                " Dy, Gd, Eu and Sm) oxides by X-ray and neutron diffraction},"
                "author = {Salinas-Sanchez, A. and Garcia-Muñoz, J.L. and Rodriguez-Carvajal, "
                "J. and Saez-Puche, R. and Martinez, J.L.},"
                "journal = {Journal of Solid State Chemistry},"
                "volume = {100},"
                "issue = {2},"
                "pages = {201-211},"
                "year = {1992},"
                "doi = {10.1016/0022-4596(92)90094-C},"
                "url = {https://doi.org/10.1016/0022-4596(92)90094-C}}",
                ]


class StructuralComplexity(BaseFeaturizer):
    """
    Shannon information entropy of a structure.

    This descriptor treat a structure as a message
    to evaluate structural complexity (:math:`S`)
    using the following equation:

    :math:`S = - v \sum_{i=1}^{k} p_i \log_2 p_i`

    :math:`p_i = m_i / v`

    where :math:`v` is the total number of atoms in the unit cell,
    :math:`p_i` is the probability mass function,
    :math:`k` is the number of symmetrically inequivalent sites, and
    :math:`m_i` is the number of sites classified in :math:`i` th
    symmetrically inequivalent site.

    Features:
        - information entropy (bits/atom)
        - information entropy (bits/unit cell)

    Args:
        symprec: precision for symmetrizing a structure
    """

    def __init__(self, symprec=0.1):
        self.symprec = symprec

    def featurize(self, struct):
        n_of_atoms = len(struct.sites)

        sga = SpacegroupAnalyzer(struct, symprec=self.symprec)
        sym_s = sga.get_symmetrized_structure()

        v = n_of_atoms
        iG = 0

        for eq_site in sym_s.equivalent_sites:
            m_i = len(eq_site)
            p_i = m_i / v
            iG -= p_i * np.log2(p_i)

        iG_total = iG * n_of_atoms

        return(iG, iG_total)

    def implementors(self):
        return ["Koki Muraoka"]

    def feature_labels(self):
        return [
            "structural complexity per atom",
            "structural complexity per cell"
        ]

    def citations(self):
        return [
            "@article{complexity2013,"
            "author = {Krivovichev, S. V.},"
            "title = {Structural complexity of minerals: information storage and processing in the mineral world},"
            "journal = {Mineral. Mag.},"
            "volume = {77},"
            "number = {3},"
            "pages = {275-326},"
            "year = {2013},"
            "month = {04},"
            "issn = {0026-461X},"
            "doi = {10.1180/minmag.2013.077.3.05},"
            "url = {https://doi.org/10.1180/minmag.2013.077.3.05}}",
        ]


class EwaldEnergy(BaseFeaturizer):
    """
    Compute the energy from Coulombic interactions.

    Note: The energy is computed using _charges already defined for the structure_.

    Features:
        ewald_energy - Coulomb interaction energy of the structure"""

    def __init__(self, accuracy=4):
        """
        Args:
            accuracy (int): Accuracy of Ewald summation, number of decimal places
        """
        self.accuracy = accuracy

    def featurize(self, strc):
        """

        Args:
             (Structure) - Structure being analyzed
        Returns:
            ([float]) - Electrostatic energy of the structure
        """
        # Compute the total energy
        ewald = EwaldSummation(strc, acc_factor=self.accuracy)
        return [ewald.total_energy]

    def feature_labels(self):
        return ["ewald_energy"]

    def implementors(self):
        return ["Logan Ward"]

    def citations(self):
        return ["@Article{Ewald1921,"
                "author = {Ewald, P. P.},"
                "doi = {10.1002/andp.19213690304},"
                "issn = {00033804},"
                "journal = {Annalen der Physik},"
                "number = {3},"
                "pages = {253--287},"
                "title = {{Die Berechnung optischer und elektrostatischer "
                "Gitterpotentiale}},"
                "url = {http://doi.wiley.com/10.1002/andp.19213690304},"
                "volume = {369},"
                "year = {1921}"
                "}"]


class MinimumRelativeDistances(BaseFeaturizer):
    """
    Determines the relative distance of each site to its closest neighbor.

    We use the relative distance,
    f_ij = r_ij / (r^atom_i + r^atom_j), as a measure rather than the
    absolute distances, r_ij, to account for the fact that different
    atoms/species have different sizes.  The function uses the
    valence-ionic radius estimator implemented in Pymatgen.
    Args:
        cutoff: (float) (absolute) distance up to which tentative
                closest neighbors (on the basis of relative distances)
                are to be determined.
    """

    def __init__(self, cutoff=10.0):
        self.cutoff = cutoff

    def featurize(self, s, cutoff=10.0):
        """
        Get minimum relative distances of all sites of the input structure.

        Args:
            s: Pymatgen Structure object.

        Returns:
            dists_relative_min: (list of floats) list of all minimum relative
                    distances (i.e., for all sites).
        """
        vire = ValenceIonicRadiusEvaluator(s)
        dists_relative_min = []
        for site in vire.structure:
            dists_relative = []
            for nnsite, dist, *_ in vire.structure.get_neighbors(site, self.cutoff):
                r_site = vire.radii[site.species_string]
                r_neigh = vire.radii[nnsite.species_string]
                radii_dist = r_site + r_neigh
                d_relative = dist / radii_dist
                dists_relative.append(d_relative)
            dists_relative_min.append(min(dists_relative))
        return [dists_relative_min]

    def feature_labels(self):
        return ["minimum relative distance of each site"]

    def citations(self):
        return ["@article{Zimmermann2017,"
                "author = {Zimmermann, Nils E. R. and Horton, Matthew K."
                " and Jain, Anubhav and Haranczyk, Maciej},"
                "doi = {10.3389/fmats.2017.00034},"
                "journal = {Frontiers in Materials},"
                "pages = {34},"
                "title = {{Assessing Local Structure Motifs Using Order"
                " Parameters for Motif Recognition, Interstitial"
                " Identification, and Diffusion Path Characterization}},"
                "url = {https://www.frontiersin.org/articles/10.3389/fmats.2017.00034},"
                "volume = {4},"
                "year = {2017}"
                "}"]

    def implementors(self):
        return ["Nils E. R. Zimmermann", "Alex Dunn"]
