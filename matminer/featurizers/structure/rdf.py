"""
Structure featurizers implementing radial distribution functions.
"""
import math
import itertools
from operator import itemgetter
from copy import copy

import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.local_env import ValenceIonicRadiusEvaluator
from pymatgen.core.periodic_table import Specie, Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from matminer.featurizers.base import BaseFeaturizer
from matminer.featurizers.utils.oxidation import has_oxidation_states


class RadialDistributionFunction(BaseFeaturizer):
    """
    Calculate the radial distribution function (RDF) of a crystal structure.

    Features:
        - Radial distribution function. Each feature is the "density" of the
          distribution at a certain radius.

    Args:
        cutoff: (float) Angstrom distance up to which to calculate the RDF.
        bin_size: (float) size in Angstrom of each bin of the (discrete) RDF.

    Attributes:
        bin_distances (np.Ndarray): The distances each bin represents. Can be
            used for graphing the RDF.
    """

    def __init__(self, cutoff=20.0, bin_size=0.1):
        self.cutoff = cutoff
        self.bin_size = bin_size
        self.bin_distances = np.arange(0, cutoff, bin_size)

    def precheck(self, s):
        """
        Precheck the structure is ordered.
        Args:
            s: (pymatgen.Struture)
        Returns:
            (bool): True if passing precheck, false if failing
        """
        return s.is_ordered

    def featurize(self, s):
        """
        Get RDF of the input structure.
        Args:
            s (Structure): Pymatgen Structure object.

        Returns:
            rdf: (iterable) the first element is the
                    normalized RDF, whereas the second element is
                    the inner radius of the RDF bin.
        """
        if not s.is_ordered:
            raise ValueError("Disordered structure support not built yet")

        # Get the distances between all atoms
        neighbors_lst = s.get_all_neighbors(self.cutoff)
        all_distances = np.concatenate(tuple(map(lambda x: [itemgetter(1)(e) for e in x], neighbors_lst)))

        # Compute a histogram
        dist_hist, dist_bins = np.histogram(
            all_distances,
            bins=np.arange(0, self.cutoff + self.bin_size, self.bin_size),
            density=False,
        )

        # Normalize counts
        shell_vol = 4.0 / 3.0 * math.pi * (np.power(dist_bins[1:], 3) - np.power(dist_bins[:-1], 3))
        number_density = s.num_sites / s.volume
        rdf = dist_hist / shell_vol / number_density
        return rdf

    def feature_labels(self):
        bin_labels = get_rdf_bin_labels(self.bin_distances, self.cutoff)
        bin_labels = [f"rdf {bl}A" for bl in bin_labels]
        return bin_labels

    def citations(self):
        return []

    def implementors(self):
        return ["Saurabh Bajaj", "Alex Dunn"]


class PartialRadialDistributionFunction(BaseFeaturizer):
    """
    Compute the partial radial distribution function (PRDF) of an xtal structure

    The PRDF of a crystal structure is the radial distibution function broken
    down for each pair of atom types.  The PRDF was proposed as a structural
    descriptor by [Schutt *et al.*]
    (https://journals.aps.org/prb/abstract/10.1103/PhysRevB.89.205118)

    Args:
        cutoff: (float) distance up to which to calculate the RDF.
        bin_size: (float) size of each bin of the (discrete) RDF.
        include_elems: (list of string), list of elements that must be included in PRDF
        exclude_elems: (list of string), list of elmeents that should not be included in PRDF

    Features:
        Each feature corresponds to the density of number of bonds
        for a certain pair of elements at a certain range of
        distances. For example, "Al-Al PRDF r=1.00-1.50" corresponds
        to the density of Al-Al bonds between 1 and 1.5 distance units
        By default, this featurizer generates RDFs for each pair
        of elements in the training set.

    """

    def __init__(self, cutoff=20.0, bin_size=0.1, include_elems=(), exclude_elems=()):
        self.cutoff = cutoff
        self.bin_size = bin_size
        self.elements_ = None
        self.include_elems = list(include_elems)  # Makes sure the element lists are ordered
        self.exclude_elems = list(exclude_elems)

    def precheck(self, s):
        """
        Precheck the structure is ordered.
        Args:
            s: (pymatgen.Struture)
        Returns:
            (bool): True if passing precheck, false if failing
        """
        return s.is_ordered

    def fit(self, X, y=None):
        """Define the list of elements to be included in the PRDF. By default,
        the PRDF will include all of the elements in `X`

        Args:
            X: (numpy array nx1) structures used in the training set. Each entry
                must be Pymatgen Structure objects.
            y: *Not used*
            fit_kwargs: *not used*

        Returns:
            self
        """

        # Initialize list with included elements
        elements = set([Element(e) for e in self.include_elems])

        # Get all of elements that appaer
        for strc in X:
            elements.update([e.element if isinstance(e, Specie) else e for e in strc.composition.keys()])

        # Remove the elements excluded by the user
        elements.difference_update([Element(e) for e in self.exclude_elems])

        # Store the elements
        self.elements_ = [e.symbol for e in sorted(elements)]

        return self

    def featurize(self, s):
        """
        Get PRDF of the input structure.
        Args:
            s: Pymatgen Structure object.

        Returns:
            prdf, dist: (tuple of arrays) the first element is a
                    dictionary where keys are tuples of element
                    names and values are PRDFs.
        """

        if not s.is_ordered:
            raise ValueError("Disordered structure support not built yet")
        if self.elements_ is None:
            raise Exception("You must run 'fit' first!")

        dist_bins, prdf = self.compute_prdf(s)  # Assemble the PRDF for each pair

        # Convert the PRDF into a feature array
        zeros = np.zeros_like(dist_bins)  # Zeros if elements don't appear
        output = []
        for key in itertools.combinations_with_replacement(self.elements_, 2):
            output.append(prdf.get(key, zeros))

        # Stack them together
        return np.hstack(output)

    def compute_prdf(self, s):
        """Compute the PRDF for a structure

        Args:
            s: (Structure), structure to be evaluated
        Returns:
            dist_bins - float, start of each of the bins
            prdf - dict, where the keys is a pair of elements (strings),
                and the value is the radial distribution function for those paris of elements
        """
        # Get the composition of the array
        s = copy(s)
        s.remove_oxidation_states()
        composition = s.composition.fractional_composition.to_reduced_dict

        # Get the distances between all atoms
        neighbors_lst = s.get_all_neighbors(self.cutoff)

        # Sort neighbors by type
        distances_by_type = {}
        for p in itertools.product(composition.keys(), composition.keys()):
            distances_by_type[p] = []

        def get_symbol(site):
            return site.specie.symbol if isinstance(site.specie, Element) else site.specie.element.symbol

        for site, nlst in zip(s.sites, neighbors_lst):  # Each list is a list for each site
            my_elem = get_symbol(site)

            for neighbor in nlst:
                rij = neighbor[1]
                n_elem = get_symbol(neighbor[0])
                # LW 3May17: Any better ideas than appending each element at a time?
                distances_by_type[(my_elem, n_elem)].append(rij)

        # Compute and normalize the prdfs
        prdf = {}
        dist_bins = self._make_bins()
        shell_volume = 4.0 / 3.0 * math.pi * (np.power(dist_bins[1:], 3) - np.power(dist_bins[:-1], 3))
        for key, distances in distances_by_type.items():
            # Compute histogram of distances
            dist_hist, dist_bins = np.histogram(distances, bins=dist_bins, density=False)
            # Normalize
            n_alpha = composition[key[0]] * s.num_sites
            rdf = dist_hist / shell_volume / n_alpha

            prdf[key] = rdf

        return dist_bins[:-1], prdf

    def _make_bins(self):
        """Generate the edges of the bins for the PRDF

        Returns:
            [list of float], edges of the bins
        """
        return np.arange(0, self.cutoff + self.bin_size, self.bin_size)

    def feature_labels(self):
        if self.elements_ is None:
            raise Exception("You must run 'fit' first!")
        bin_edges = self._make_bins()
        labels = []
        for e1, e2 in itertools.combinations_with_replacement(self.elements_, 2):
            for r_start, r_end in zip(bin_edges, bin_edges[1:]):
                labels.append("{}-{} PRDF r={:.2f}-{:.2f}".format(e1, e2, r_start, r_end))
        return labels

    def citations(self):
        return [
            "@article{Schutt2014,"
            'author = {Sch{"{u}}tt, K. T. and Glawe, H. and Brockherde, F. '
            'and Sanna, A. and M{"{u}}ller, K. R. and Gross, E. K. U.},'
            "doi = {10.1103/PhysRevB.89.205118},"
            "journal = {Physical Review B},"
            "month = {may},number = {20},pages = {205118},"
            "title = {{How to represent crystal structures for machine learning:"
            " Towards fast prediction of electronic properties}},"
            "url = {http://link.aps.org/doi/10.1103/PhysRevB.89.205118},"
            "volume = {89},"
            "year = {2014}}"
        ]

    def implementors(self):
        return ["Logan Ward", "Saurabh Bajaj"]


class ElectronicRadialDistributionFunction(BaseFeaturizer):
    """
    Calculate the inherent electronic radial distribution function (ReDF)

    The ReDF is defined according to Willighagen et al., Acta Cryst., 2005, B61,
    29-36.

    The ReDF is a structure-integral RDF (i.e., summed over
    all sites) in which the positions of neighboring sites
    are weighted by electrostatic interactions inferred
    from atomic partial charges. Atomic charges are obtained
    from the ValenceIonicRadiusEvaluator class.

    WARNING: The ReDF needs oxidation states to work correctly.

    Args:
        cutoff: (float) distance up to which the ReDF is to be
                calculated.
        dr: (float) width of bins ("x"-axis) of ReDF (default: 0.05 A).

    Attributes:
        distances (np.ndarray): The distances at which each bin begins.
    """

    def __init__(self, cutoff=20, dr=0.05):
        self.cutoff = cutoff
        self.dr = dr
        self.nbins = int(self.cutoff / self.dr) + 1
        self.distances = np.array([i * self.dr for i in range(self.nbins)])

    def precheck(self, s) -> bool:
        """
        Check the structure to ensure the ReDF can be run.
        Args:
            s (pymatgen. Structure): Structure to precheck
        Returns:
            (bool)
        """
        return has_oxidation_states(s.composition) and s.is_ordered

    def featurize(self, s):
        """
        Get ReDF of input structure.

        Args:
            s: input Structure object.

        Returns: (list) the ReDF

        """

        if not has_oxidation_states(s.composition):
            raise ValueError("Structure must have oxidation states")
        if not s.is_ordered:
            raise ValueError("Structure must be ordered")
        if self.dr <= 0:
            raise ValueError("width of bins for ReDF must be >0")

        # Make structure primitive.
        struct = SpacegroupAnalyzer(s).find_primitive() or s

        # Add oxidation states.
        struct = ValenceIonicRadiusEvaluator(struct).structure

        distribution = np.zeros(self.nbins, dtype=np.float)
        nbins = int(self.cutoff / self.dr) + 1

        for site in struct.sites:
            this_charge = float(site.specie.oxi_state)
            neighbors = struct.get_neighbors(site, self.cutoff)
            for nnsite, dist, *_ in neighbors:
                neigh_charge = float(nnsite.specie.oxi_state)
                bin_index = int(dist / self.dr)
                distribution[bin_index] += (this_charge * neigh_charge) / (struct.num_sites * dist)

        return distribution

    def feature_labels(self):
        bin_labels = get_rdf_bin_labels(self.distances, self.cutoff)
        bin_labels = [f"ReDF {bl}A" for bl in bin_labels]
        return bin_labels

    def citations(self):
        return [
            "@article{title={Method for the computational comparison"
            " of crystal structures}, volume={B61}, pages={29-36},"
            " DOI={10.1107/S0108768104028344},"
            " journal={Acta Crystallographica Section B},"
            " author={Willighagen, E. L. and Wehrens, R. and Verwer,"
            " P. and de Gelder R. and Buydens, L. M. C.}, year={2005}}"
        ]

    def implementors(self):
        return ["Nils E. R. Zimmermann", "Alex Dunn"]


def get_rdf_bin_labels(bin_distances, cutoff):
    """
    Common function for getting bin labels given the distances at which each
    bin begins and the ending cutoff.
    Args:
        bin_distances (np.ndarray): The distances at which each bin begins.
        cutoff (float): The final cutoff value.
    Returns:
        [str]: The feature labels for the *RDF
    """
    bin_dists_complete = np.concatenate((bin_distances, np.asarray([cutoff])))
    flabels = [""] * len(bin_distances)
    for i, _ in enumerate(bin_distances):
        lower = "{:.5f}".format(bin_dists_complete[i])
        higher = "{:.5f}".format(bin_dists_complete[i + 1])
        flabels[i] = f"[{lower} - {higher}]"
    return flabels
