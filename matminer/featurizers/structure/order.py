"""
Structure featurizers based on packing or ordering.
"""

import math

import numpy as np
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from matminer.featurizers.base import BaseFeaturizer
from matminer.utils.caching import get_all_nearest_neighbors


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
        self.features = ["density", "vpa", "packing fraction"] if not desired_features else desired_features

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


class ChemicalOrdering(BaseFeaturizer):
    """
    How much the ordering of species in the structure differs from random

    These parameters describe how much the ordering of all species in a
    structure deviates from random using a Warren-Cowley-like ordering
    parameter. The first step of this calculation is to determine the nearest
    neighbor shells of each site. Then, for each shell a degree of order for
    each type is determined by computing:

    :math:`\\alpha (t,s) = 1 - \\frac{\\sum_n w_n \\delta (t - t_n)}{x_t \\sum_n w_n}`

    where :math:`w_n` is the weight associated with a certain neighbor,
    :math:`t_p` is the type of the neighbor, and :math:`x_t` is the fraction
    of type t in the structure. For atoms that are randomly dispersed in a
    structure, this formula yields 0 for all types. For structures where
    each site is surrounded only by atoms of another type, this formula
    yields large values of :math:`alpha`.

    The mean absolute value of this parameter across all sites is used
    as a feature.

    Features:
        mean ordering parameter shell [n] - Mean ordering parameter for
            atoms in the n<sup>th</sup> neighbor shell

    References:
         `Ward et al. _PRB_ 2017 <http://link.aps.org/doi/10.1103/PhysRevB.96.024104>`_"""

    def __init__(self, shells=(1, 2, 3), weight="area"):
        """Initialize the featurizer

        Args:
            shells ([int]) - Which neighbor shells to evaluate
            weight (str) - Attribute used to weigh neighbor contributions
        """
        self.shells = shells
        self.weight = weight

    def featurize(self, strc):
        # Shortcut: Return 0 if there is only 1 type of atom
        if len(strc.composition) == 1:
            return [0] * len(self.shells)

        # Get a list of types
        elems, fracs = zip(*strc.composition.element_composition.fractional_composition.items())

        # Precompute the list of NNs in the structure
        voro = VoronoiNN(weight=self.weight)
        all_nn = get_all_nearest_neighbors(voro, strc)

        # Evaluate each shell
        output = []
        for shell in self.shells:
            # Initialize an array to store the ordering parameters
            ordering = np.zeros((len(strc), len(elems)))

            # Get the ordering of each type of each atom
            for site_idx in range(len(strc)):
                nns = voro._get_nn_shell_info(strc, all_nn, site_idx, shell)

                # Sum up the weights
                total_weight = sum(x["weight"] for x in nns)

                # Get weight by type
                for nn in nns:
                    site_elem = nn["site"].specie
                    if hasattr(site_elem, "element"):
                        site_elem = getattr(site_elem, "element")

                    elem_idx = elems.index(site_elem)
                    ordering[site_idx, elem_idx] += nn["weight"]

                # Compute the ordering parameter
                ordering[site_idx, :] = 1 - ordering[site_idx, :] / total_weight / np.array(fracs)

            # Compute the average ordering for the entire structure
            output.append(np.abs(ordering).mean())

        return output

    def feature_labels(self):
        return [f"mean ordering parameter shell {n}" for n in self.shells]

    def citations(self):
        return [
            "@article{Ward2017,"
            "author = {Ward, Logan and Liu, Ruoqian "
            "and Krishna, Amar and Hegde, Vinay I. "
            "and Agrawal, Ankit and Choudhary, Alok "
            "and Wolverton, Chris},"
            "doi = {10.1103/PhysRevB.96.024104},"
            "journal = {Physical Review B},"
            "pages = {024104},"
            "title = {{Including crystal structure attributes "
            "in machine learning models of formation energies "
            "via Voronoi tessellations}},"
            "url = {http://link.aps.org/doi/10.1103/PhysRevB.96.024104},"
            "volume = {96},year = {2017}}"
        ]

    def implementors(self):
        return ["Logan Ward"]


class MaximumPackingEfficiency(BaseFeaturizer):
    """
    Maximum possible packing efficiency of this structure

    Uses a Voronoi tessellation to determine the largest radius each atom
    can have before any atoms touches any one of their neighbors. Given the
    maximum radius size, this class computes the maximum packing efficiency
    of the structure as a feature.

    Features:
        max packing efficiency - Maximum possible packing efficiency
    """

    def featurize(self, strc):
        # Get the Voronoi tessellation of each site
        voro = VoronoiNN()
        nns = [voro.get_voronoi_polyhedra(strc, i) for i in range(len(strc))]

        # Compute the radius of largest possible atom for each site
        #  The largest radius is equal to the distance from the center of the
        #   cell to the closest Voronoi face
        max_r = [min(x["face_dist"] for x in nn.values()) for nn in nns]

        # Compute the packing efficiency
        return [4.0 / 3.0 * np.pi * np.power(max_r, 3).sum() / strc.volume]

    def feature_labels(self):
        return ["max packing efficiency"]

    def citations(self):
        return [
            "@article{Ward2017,"
            "author = {Ward, Logan and Liu, Ruoqian "
            "and Krishna, Amar and Hegde, Vinay I. "
            "and Agrawal, Ankit and Choudhary, Alok "
            "and Wolverton, Chris},"
            "doi = {10.1103/PhysRevB.96.024104},"
            "journal = {Physical Review B},"
            "pages = {024104},"
            "title = {{Including crystal structure attributes "
            "in machine learning models of formation energies "
            "via Voronoi tessellations}},"
            "url = {http://link.aps.org/doi/10.1103/PhysRevB.96.024104},"
            "volume = {96},year = {2017}}"
        ]

    def implementors(self):
        return ["Logan Ward"]


class StructuralComplexity(BaseFeaturizer):
    r"""
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

        return (iG, iG_total)

    def implementors(self):
        return ["Koki Muraoka"]

    def feature_labels(self):
        return ["structural complexity per atom", "structural complexity per cell"]

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
