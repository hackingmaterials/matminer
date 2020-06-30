from __future__ import division, unicode_literals, print_function

import numpy as np
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.core.periodic_table import Specie

from matminer.featurizers.base import BaseFeaturizer
from matminer.featurizers.utils.stats import PropertyStats
from matminer.utils.caching import get_all_nearest_neighbors


class StructuralHeterogeneity(BaseFeaturizer):
    """
    Variance in the bond lengths and atomic volumes in a structure

    These features are based on several statistics derived from the Voronoi
    tessellation of a structure. The first set of features relate to the
    variance in the average bond length across all atoms in the structure.
    The second relate to the variance of bond lengths between each neighbor
    of each atom. The final feature is the variance in Voronoi cell sizes
    across the structure.

    We define the 'average bond length' of a site as the weighted average of
    the bond lengths for all neighbors. By default, the weight is the
    area of the face between the sites.

    The 'neighbor distance variation' is defined as the weighted mean absolute
    deviation in both length for all neighbors of a particular site. As before,
    the weight is according to face area by default. For this statistic, we
    divide the mean absolute deviation by the mean neighbor distance for that
    site.

    Features:
        mean absolute deviation in relative bond length - Mean absolute deviation
            in the average bond lengths for all sites, divided by the
            mean average bond length
        max relative bond length - Maximum average bond length, divided by the
            mean average bond length
        min relative bond length - Minimum average bond length, divided by the
            mean average bond length
        [stat] neighbor distance variation - Statistic (e.g., mean) of the
            neighbor distance variation
        mean absolute deviation in relative cell size - Mean absolute deviation
            in the Voronoi cell volume across all sites in the structure.
            Divided by the mean Voronoi cell volume.

    References:
         `Ward et al. _PRB_ 2017 <http://link.aps.org/doi/10.1103/PhysRevB.96.024104>`_
    """

    def __init__(self, weight='area',
                 stats=("minimum", "maximum", "range", "mean", "avg_dev")):
        self.weight = weight
        self.stats = stats

    def featurize(self, strc):
        # Compute the Voronoi tessellation of each site
        voro = VoronoiNN(extra_nn_info=True, weight=self.weight)
        nns = get_all_nearest_neighbors(voro, strc)

        # Compute the mean bond length of each atom, and the mean
        #   variation within each cell
        mean_bond_lengths = np.zeros((len(strc),))
        bond_length_var = np.zeros_like(mean_bond_lengths)
        for i, nn in enumerate(nns):
            weights = [n['weight'] for n in nn]
            lengths = [n['poly_info']['face_dist'] * 2 for n in nn]
            mean_bond_lengths[i] = PropertyStats.mean(lengths, weights)

            # Compute the mean absolute deviation of the bond lengths
            bond_length_var[i] = PropertyStats.avg_dev(lengths, weights) / \
                                 mean_bond_lengths[i]

        # Normalize the bond lengths by the average of the whole structure
        #   This is done to make the attributes length-scale-invariant
        mean_bond_lengths /= mean_bond_lengths.mean()

        # Compute statistics related to bond lengths
        features = [PropertyStats.avg_dev(mean_bond_lengths),
                    mean_bond_lengths.max(), mean_bond_lengths.min()]
        features += [PropertyStats.calc_stat(bond_length_var, stat)
                     for stat in self.stats]

        # Compute the variance in volume
        cell_volumes = [sum(x['poly_info']['volume'] for x in nn) for nn in nns]
        features.append(
            PropertyStats.avg_dev(cell_volumes) / np.mean(cell_volumes))

        return features

    def feature_labels(self):
        fl = [
            "mean absolute deviation in relative bond length",
            "max relative bond length",
            "min relative bond length"
        ]
        fl += [stat + " neighbor distance variation" for stat in self.stats]
        fl.append("mean absolute deviation in relative cell size")
        return fl

    def citations(self):
        return ["@article{Ward2017,"
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
                "volume = {96},year = {2017}}"]

    def implementors(self):
        return ['Logan Ward']


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
        max_r = [min(x['face_dist'] for x in nn.values()) for nn in nns]

        # Compute the packing efficiency
        return [4. / 3. * np.pi * np.power(max_r, 3).sum() / strc.volume]

    def feature_labels(self):
        return ['max packing efficiency']

    def citations(self):
        return ["@article{Ward2017,"
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
                "volume = {96},year = {2017}}"]

    def implementors(self):
        return ['Logan Ward']


class ChemicalOrdering(BaseFeaturizer):
    """
    How much the ordering of species in the structure differs from random

    These parameters describe how much the ordering of all species in a
    structure deviates from random using a Warren-Cowley-like ordering
    parameter. The first step of this calculation is to determine the nearest
    neighbor shells of each site. Then, for each shell a degree of order for
    each type is determined by computing:

    :math:`\\alpha (t,s) = 1 - \\frac{\sum_n w_n \delta (t - t_n)}{x_t \sum_n w_n}`

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

    def __init__(self, shells=(1, 2, 3), weight='area'):
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
        elems, fracs = zip(
            *strc.composition.element_composition.fractional_composition.items())

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
                total_weight = sum(x['weight'] for x in nns)

                # Get weight by type
                for nn in nns:
                    site_elem = nn['site'].specie.element \
                        if isinstance(nn['site'].specie, Specie) else \
                        nn['site'].specie
                    elem_idx = elems.index(site_elem)
                    ordering[site_idx, elem_idx] += nn['weight']

                # Compute the ordering parameter
                ordering[site_idx, :] = 1 - ordering[site_idx, :] / \
                                        total_weight / np.array(fracs)

            # Compute the average ordering for the entire structure
            output.append(np.abs(ordering).mean())

        return output

    def feature_labels(self):
        return ["mean ordering parameter shell {}".format(n) for n in
                self.shells]

    def citations(self):
        return ["@article{Ward2017,"
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
                "volume = {96},year = {2017}}"]

    def implementors(self):
        return ['Logan Ward']
