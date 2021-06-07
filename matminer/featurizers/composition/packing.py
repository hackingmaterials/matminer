"""
Composition featurizers for determining packing characteristics.
"""

import itertools
from functools import lru_cache

import numpy as np
from pymatgen.core.composition import Composition
from sklearn.neighbors import NearestNeighbors

from matminer.featurizers.base import BaseFeaturizer
from matminer.featurizers.utils.stats import PropertyStats
from matminer.utils.data import (
    MagpieData,
)
from matminer.featurizers.composition.element import ElementFraction


class AtomicPackingEfficiency(BaseFeaturizer):
    """
    Packing efficiency based on a geometric theory of the amorphous packing
    of hard spheres.

    This featurizer computes two different kinds of the features. The first
    relate to the distance between a composition and the composition of
    the clusters of atoms expected to be efficiently packed based on a
    theory from
    `Laws et al.<http://www.nature.com/doifinder/10.1038/ncomms9123>`_.
    The second corresponds to the packing efficiency of a system if all atoms
    in the alloy are simultaneously as efficiently-packed as possible.

    The packing efficiency in these models is based on the Atomic Packing
    Efficiency (APE), which measures the difference between the ratio of
    the radii of the central atom to its neighbors and the ideal ratio
    of a cluster with the same number of atoms that has optimal packing
    efficiency. If the difference between the ratios is too large, the APE is
    positive. If the difference is too small, the APE is negative.

    Features:
        dist from {k} clusters |APE| < {thr} - The distance between an
            alloy composition and the k clusters that have a packing efficiency
            below thr from ideal
        mean simul. packing efficiency - Mean packing efficiency of all atoms.
            The packing efficiency is measured with respect to ideal (0)
        mean abs simul. packing efficiency - Mean absolute value of the
            packing efficiencies. Closer to zero is more efficiently packed

    References:
        [1] K.J. Laws, D.B. Miracle, M. Ferry, A predictive structural model
        for bulk metallic glasses, Nat. Commun. 6 (2015) 8123. doi:10.1038/ncomms9123.
    """

    def __init__(self, threshold=0.01, n_nearest=(1, 3, 5), max_types=6):
        """
        Initialize the featurizer

        Args:
            threshold (float):Threshold to use for determining whether
                a cluster is efficiently packed.
            n_nearest ({int}): Number of nearest clusters to use when considering features
            max_types (int): Maximum number of atom types to consider when
                looking for efficient clusters. The process for finding
                efficient clusters very expensive for large numbers of types
        """

        # Store the options
        self.threshold = threshold
        self.n_nearest = n_nearest
        self.max_types = max_types

        # Tool to convert composition objects to fractions as a vector
        self._el_frac = ElementFraction()

        # Get the number of elements in the output of `_el_frac`
        self._n_elems = len(self._el_frac.featurize(Composition("H")))

        # Tool for looking up radii
        self._data_source = MagpieData()

        # Lookup table of ideal radius ratios
        self.ideal_ratio = dict(
            [
                (3, 0.154701),
                (4, 0.224745),
                (5, 0.361654),
                (6, 0.414214),
                (7, 0.518145),
                (8, 0.616517),
                (9, 0.709914),
                (10, 0.798907),
                (11, 0.884003),
                (12, 0.902113),
                (13, 0.976006),
                (14, 1.04733),
                (15, 1.11632),
                (16, 1.18318),
                (17, 1.2481),
                (18, 1.31123),
                (19, 1.37271),
                (20, 1.43267),
                (21, 1.49119),
                (22, 1.5484),
                (23, 1.60436),
                (24, 1.65915),
            ]
        )

    def __hash__(self):
        return hash(self.threshold)

    def __eq__(self, other):
        if isinstance(other, AtomicPackingEfficiency):
            return self.get_params() == other.get_params()

    def featurize(self, comp):
        return list(self.compute_simultaneous_packing_efficiency(comp)) + self.compute_nearest_cluster_distance(comp)

    def feature_labels(self):
        return [
            "mean simul. packing efficiency",
            "mean abs simul. packing efficiency",
        ] + ["dist from {} clusters |APE| < {:.3f}".format(k, self.threshold) for k in self.n_nearest]

    def citations(self):
        return [
            "@article{Laws2015,"
            "author = {Laws, K. J. and Miracle, D. B. and Ferry, M.},"
            "doi = {10.1038/ncomms9123},"
            "journal = {Nature Communications},"
            "pages = {8123},"
            "title = {{A predictive structural model for bulk metallic glasses}},"
            "url = {http://www.nature.com/doifinder/10.1038/ncomms9123},"
            "volume = {6},"
            "year = {2015}"
        ]

    def implementors(self):
        return ["Logan Ward"]

    def compute_simultaneous_packing_efficiency(self, comp):
        """Compute the packing efficiency of the system when the neighbor
        shell of each atom has the same composition as the alloy. When this
        criterion is satisfied, it is possible for every atom in this system
        to be simultaneously as efficiently-packed as possible.

        Args:
            comp (Composition): Composition to be assessed
        Returns
            (float) Average APE of all atoms
            (float) Average deviation of the APE of each atom from ideal (0)
        """

        # Compute the average atomic radius of the system
        elements, fractions = zip(*comp.element_composition.items())
        radii = self._data_source.get_elemental_properties(elements, "MiracleRadius")
        mean_radius = PropertyStats.mean(radii, fractions)

        # Compute the APE for each cluster
        best_ape = [self.find_ideal_cluster_size(r / mean_radius)[1] for r in radii]

        # Return the averages
        return PropertyStats.mean(best_ape, fractions), PropertyStats.mean(np.abs(best_ape), fractions)

    def compute_nearest_cluster_distance(self, comp):
        """Compute the distance between a composition and that the nearest
        efficiently-packed clusters.

        Measures the mean :math:`L_2` distance between the alloy composition
        and the :math:`k`-nearest clusters with Atomic Packing Efficiencies
        within the user-specified tolerance of 1. :math:`k` is any of the
        numbers defined in the "n_nearest" parameter of this class.

        If there are less than `k` efficient clusters in the system, we use
        the maximum distance betweeen any two compositions (1) for the
        unmatched neighbors.

        Args:
            comp (Composition): Composition of material to evaluate
        Return:
            [float] Average distances
        """

        # Get the most common elements
        elems, _ = zip(*sorted(comp.element_composition.items(), key=lambda x: x[1], reverse=True))

        # Get the cluster lookup tool using the most common elements
        cluster_lookup = self.create_cluster_lookup_tool(elems[: self.max_types])

        # Compute the composition vector
        comp_vec = self._el_frac.featurize(comp)

        # Compute the distances
        means = []
        for k in self.n_nearest:
            # Get the nearest clusters
            if cluster_lookup is None:
                dists = (np.array([]),)
                to_lookup = 0
            else:
                to_lookup = min(cluster_lookup._fit_X.shape[0], k)
                dists, _ = cluster_lookup.kneighbors([comp_vec], to_lookup)

            # Pad the list with 1's
            dists = dists[0].tolist() + [1] * (k - to_lookup)

            # Compute the average
            means.append(np.mean(dists))

        return means

    def create_cluster_lookup_tool(self, elements):
        """
        Get the compositions of efficiently-packed clusters in a certain system
        of elements

        Args:
            elements ([Element]): Elements in system
        Return:
            (NearNeighbors): Tool to find nearby clusters in this system. None
                if there are no efficiently-packed clusters for this combination of elements
        """
        elements = list(set(elements))
        return self._create_cluster_lookup_tool(tuple(sorted(elements)))

    @lru_cache()
    def _create_cluster_lookup_tool(self, elements):
        """
        Cached version of `create_cluster_lookup_tool`. Assumes that the
        elements are passed as sorted tuple with no duplicates

        Args:
            elements ([Element]): Elements in system
        Return:
            (NearNeighbors): Tool to find nearby clusters in this system. If
            there are no clusters, this class returns None
        """

        # Get the radii
        radii = self._data_source.get_elemental_properties(elements, "MiracleRadius")

        # Get the maximum and minimum cluster sizes
        max_size = self.find_ideal_cluster_size(max(radii) / min(radii))[0]
        min_size = self.find_ideal_cluster_size(min(radii) / max(radii))[0]

        # Prepare a list to hold all possible clusters
        eff_clusters = []

        # Loop through all possible neighbor shells
        for size in range(min_size, max_size + 1):
            # Get the ideal radius ratio for a cluster of this size
            ideal_ratio = self.get_ideal_radius_ratio(size)

            # Get the mean radii and compositions of all possible
            #  combinations of elements in the neighbor shell
            s_radii = itertools.combinations_with_replacement(radii, size)
            s_elems = itertools.combinations_with_replacement(elements, size)

            #  Put the results in arrays for fast indexing
            mean_radii = np.array(list(s_radii)).mean(axis=1)
            s_elems = np.array(list(s_elems))

            # For each type of central atom, determine which have an APE
            #  within `self.threshold` of 1
            for center_radius, center_elem in zip(radii, elements):
                # Compute the APE of each cluster
                ape = 1 - np.divide(ideal_ratio, np.divide(center_radius, mean_radii))

                # Get those which are within the threshold of 0
                #  and add their composition to the list of OK elements
                for hit in s_elems[np.abs(ape) < self.threshold]:
                    eff_clusters.append([center_elem] + hit.tolist())

        # Compute the composition vectors for all of the efficient clusters
        comps = np.zeros((len(eff_clusters), self._n_elems))
        for i, elems in enumerate(eff_clusters):
            for elem in elems:
                comps[i, elem.Z - 1] += 1
        comps = np.divide(comps, comps.sum(axis=1)[:, None])

        # Return tool to quickly determine distance from efficient clusters
        #  NearNeighbors requires at least 1 entry, so we return None if
        #   there are no nearby clusters
        return NearestNeighbors().fit(comps) if len(comps) > 0 else None

    def find_ideal_cluster_size(self, radius_ratio):
        """
        Get the optimal cluster size for a certain radius ratio

        Finds the number of nearest neighbors :math:`n` that minimizes
        :math:`|1 - rp(n)/r|`, where :math:`rp(n)` is the ideal radius
        ratio for a certain :math:`n` and :math:`r` is the actual ratio.

        Args:
            radius_ratio (float): :math:`r / r_{neighbor}`
        Returns:
            (int) number of neighboring atoms for that will be the most
            efficiently packed.
            (float) Optimal APE
        """

        # Loop through cluster sizes from 3 to 24
        best_ape = np.inf
        best_n = None
        for n in range(3, 25):
            # Compute APE, check if it is the best
            ape = 1 - self.get_ideal_radius_ratio(n) / radius_ratio
            if abs(ape) < abs(best_ape):
                best_ape = ape
                best_n = n

            # If the APE is negative, this is either the best APE or
            #  We have already passed it
            if ape < 0:
                return best_n, best_ape

        return best_n, best_ape

    def get_ideal_radius_ratio(self, n_neighbors):
        """Compute the idea ratio between the central atom and neighboring
        atoms for a neighbor with a certain number of nearest neighbors.

        Based on work by `Miracle, Lord, and Ranganathan
        <https://www.jstage.jst.go.jp/article/matertrans/47/7/47_7_1737/_article/-char/en>`_.

        Args:
            n_neighbors (int): Number of atoms in 1st NN shell
        Return:
            (float) ideal radius ratio :math:`r / r_{neighbor}`
        """

        # NN must be in [3, 24]
        n = max(3, min(n_neighbors, 24))

        return self.ideal_ratio[n]
