from __future__ import division, unicode_literals, print_function

import itertools
import warnings
from collections import OrderedDict
from math import pi
from operator import itemgetter

import numpy as np
import pandas as pd
import pymatgen.analysis.local_env as pmg_le
import scipy.constants as const
from pymatgen import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.analysis.local_env import ValenceIonicRadiusEvaluator
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.analysis.structure_analyzer import get_dimensionality
from pymatgen.core.periodic_table import Specie, Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.stats import gaussian_kde
from sklearn.exceptions import NotFittedError

from matminer.featurizers.base import BaseFeaturizer
from matminer.featurizers.site import OPSiteFingerprint, \
    CoordinationNumber, LocalPropertyDifference, CrystalNNFingerprint, \
    AverageBondAngle, AverageBondLength
from matminer.featurizers.utils.stats import PropertyStats
from matminer.utils.caching import get_all_nearest_neighbors

__authors__ = 'Anubhav Jain <ajain@lbl.gov>, Saurabh Bajaj <sbajaj@lbl.gov>, '\
              'Nils E.R. Zimmerman <nils.e.r.zimmermann@gmail.com>, ' \
              'Alex Dunn <ardunn@lbl.gov>'

ANG_TO_BOHR = const.value('Angstrom star') / const.value('Bohr radius')


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
            output.append(4 * pi * total_rad / (3 * s.volume))

        return output

    def feature_labels(self):
        all_features = ["density", "vpa", "packing fraction"]  # enforce order
        return [x for x in all_features if x in self.features]

    def citations(self):
        return []

    def implementors(self):
        return ["Saurabh Bajaj", "Anubhav Jain"]


class GlobalSymmetryFeatures(BaseFeaturizer):
    """
    Determines symmetry features, e.g. spacegroup number and  crystal system

    Features:
        - Spacegroup number
        - Crystal system (1 of 7)
        - Centrosymmetry (has inversion symmetry)
    """

    crystal_idx = {"triclinic": 7,
                   "monoclinic": 6,
                   "orthorhombic": 5,
                   "tetragonal": 4,
                   "trigonal": 3,
                   "hexagonal": 2,
                   "cubic": 1
                   }

    def __init__(self, desired_features=None):
        self.features = ["spacegroup_num", "crystal_system",
                         "crystal_system_int", "is_centrosymmetric"] if not \
            desired_features else desired_features

    def featurize(self, s):
        sga = SpacegroupAnalyzer(s)
        output = []

        if "spacegroup_num" in self.features:
            output.append(sga.get_space_group_number())

        if "crystal_system" in self.features:
            output.append(sga.get_crystal_system())

        if "crystal_system_int" in self.features:
            output.append(GlobalSymmetryFeatures.crystal_idx[
                              sga.get_crystal_system()])

        if "is_centrosymmetric" in self.features:
            output.append(sga.is_laue())

        return output

    def feature_labels(self):
        all_features = ["spacegroup_num", "crystal_system",
                        "crystal_system_int",
                        "is_centrosymmetric"]  # enforce order
        return [x for x in all_features if x in self.features]

    def citations(self):
        return []

    def implementors(self):
        return ["Anubhav Jain"]


class Dimensionality(BaseFeaturizer):
    """
    Returns dimensionality of structure: 1 means linear chains of atoms OR
    isolated atoms/no bonds, 2 means layered, 3 means 3D connected
    structure. This feature is sensitive to bond length tables that you use.
    """

    def __init__(self, **kwargs):
        """

        Args:
            **kwargs: keyword args to pass to get_dimensionality() method of
                pymatgen.
        """
        self.kwargs = kwargs

    def featurize(self, s):
        return [get_dimensionality(s, **self.kwargs)]

    def feature_labels(self):
        return ["dimensionality"]

    def citations(self):
        return ["@article{Gorai2016a, "
                "author = {Gorai, Prashun and Toberer, Eric and Stevanovic, "
                "Vladan}, doi = {10.1039/C6TA04121C}, issn = {2050-7488}, "
                "journal = {J. Mater. Chem. A}, number = {12},pages = {4136}, "
                "title = {{Computational Identification of Promising "
                "Thermoelectric Materials Among Known Quasi-2D Binary "
                "Compounds}}, volume = {2}, year = {2016}}"]

    def implementors(self):
        return ["Anubhav Jain"]


class RadialDistributionFunction(BaseFeaturizer):
    """
    Calculate the radial distribution function (RDF) of a crystal structure.

    Features:
        - Radial distribution function

    Args:
        cutoff: (float) distance up to which to calculate the RDF.
        bin_size: (float) size of each bin of the (discrete) RDF.
    """

    def __init__(self, cutoff=20.0, bin_size=0.1):
        self.cutoff = cutoff
        self.bin_size = bin_size

    def featurize(self, s):
        """
        Get RDF of the input structure.
        Args:
            s (Structure): Pymatgen Structure object.

        Returns:
            rdf, dist: (tuple of arrays) the first element is the
                    normalized RDF, whereas the second element is
                    the inner radius of the RDF bin.
        """
        if not s.is_ordered:
            raise ValueError("Disordered structure support not built yet")

        # Get the distances between all atoms
        neighbors_lst = s.get_all_neighbors(self.cutoff)
        all_distances = np.concatenate(
            tuple(map(lambda x: [itemgetter(1)(e) for e in x], neighbors_lst)))

        # Compute a histogram
        dist_hist, dist_bins = np.histogram(
            all_distances, bins=np.arange(
                0, self.cutoff + self.bin_size, self.bin_size), density=False)

        # Normalize counts
        shell_vol = 4.0 / 3.0 * pi * (np.power(
            dist_bins[1:], 3) - np.power(dist_bins[:-1], 3))
        number_density = s.num_sites / s.volume
        rdf = dist_hist / shell_vol / number_density
        return [{'distances': dist_bins[:-1], 'distribution': rdf}]

    def feature_labels(self):
        return ["radial distribution function"]

    def citations(self):
        return []

    def implementors(self):
        return ["Saurabh Bajaj"]


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
           of elements in the training set."""

    def __init__(self, cutoff=20.0, bin_size=0.1, include_elems=(), exclude_elems=()):
        self.cutoff = cutoff
        self.bin_size = bin_size
        self.elements_ = None
        self.include_elems = list(include_elems)  # Makes sure the element lists are ordered
        self.exclude_elems = list(exclude_elems)

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
        composition = s.composition.fractional_composition.to_reduced_dict

        # Get the distances between all atoms
        neighbors_lst = s.get_all_neighbors(self.cutoff)

        # Sort neighbors by type
        distances_by_type = {}
        for p in itertools.product(composition.keys(), composition.keys()):
            distances_by_type[p] = []

        def get_symbol(site):
            return site.specie.symbol if isinstance(site.specie,
                                                    Element) else site.specie.element.symbol

        for site, nlst in zip(s.sites,
                              neighbors_lst):  # Each list is a list for each site
            my_elem = get_symbol(site)

            for neighbor in nlst:
                rij = neighbor[1]
                n_elem = get_symbol(neighbor[0])
                # LW 3May17: Any better ideas than appending each element at a time?
                distances_by_type[(my_elem, n_elem)].append(rij)

        # Compute and normalize the prdfs
        prdf = {}
        dist_bins = self._make_bins()
        shell_volume = 4.0 / 3.0 * pi * (
            np.power(dist_bins[1:], 3) - np.power(dist_bins[:-1], 3))
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
                labels.append("{}-{} PRDF r={:.2f}-{:.2f}".format(
                    e1, e2, r_start, r_end
                ))
        return labels

    def citations(self):
        return ["@article{Schutt2014,"
                "author = {Sch{\"{u}}tt, K. T. and Glawe, H. and Brockherde, F. "
                "and Sanna, A. and M{\"{u}}ller, K. R. and Gross, E. K. U.},"
                "doi = {10.1103/PhysRevB.89.205118},"
                "journal = {Physical Review B},"
                "month = {may},number = {20},pages = {205118},"
                "title = {{How to represent crystal structures for machine learning:"
                " Towards fast prediction of electronic properties}},"
                "url = {http://link.aps.org/doi/10.1103/PhysRevB.89.205118},"
                "volume = {89},""year = {2014}}"]

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

    Args:
        cutoff: (float) distance up to which the ReDF is to be
                calculated (default: longest diagaonal in
                primitive cell).
        dr: (float) width of bins ("x"-axis) of ReDF (default: 0.05 A).
    """

    def __init__(self, cutoff=None, dr=0.05):
        self.cutoff = cutoff
        self.dr = dr

    def featurize(self, s):
        """
        Get ReDF of input structure.

        Args:
            s: input Structure object.

        Returns: (dict) a copy of the electronic radial distribution
                functions (ReDF) as a dictionary. The distance list
                ("x"-axis values of ReDF) can be accessed via key
                'distances'; the ReDF itself is accessible via key
                'redf'.
        """
        if self.dr <= 0:
            raise ValueError("width of bins for ReDF must be >0")

        # Make structure primitive.
        struct = SpacegroupAnalyzer(s).find_primitive() or s

        # Add oxidation states.
        struct = ValenceIonicRadiusEvaluator(struct).structure

        if self.cutoff is None:
            # Set cutoff to longest diagonal.
            a = struct.lattice.matrix[0]
            b = struct.lattice.matrix[1]
            c = struct.lattice.matrix[2]
            self.cutoff = max(
                [np.linalg.norm(a + b + c), np.linalg.norm(-a + b + c),
                 np.linalg.norm(a - b + c), np.linalg.norm(a + b - c)])

        nbins = int(self.cutoff / self.dr) + 1
        redf_dict = {"distances": np.array(
            [(i + 0.5) * self.dr for i in range(nbins)]),
            "distribution": np.zeros(nbins, dtype=np.float)}

        for site in struct.sites:
            this_charge = float(site.specie.oxi_state)
            neighs_dists = struct.get_neighbors(site, self.cutoff)
            for neigh, dist in neighs_dists:
                neigh_charge = float(neigh.specie.oxi_state)
                bin_index = int(dist / self.dr)
                redf_dict["distribution"][bin_index] += (
                                                                this_charge * neigh_charge) / (
                                                                struct.num_sites * dist)

        return [redf_dict]

    def feature_labels(self):
        return ["electronic radial distribution function"]

    def citations(self):
        return ["@article{title={Method for the computational comparison"
                " of crystal structures}, volume={B61}, pages={29-36},"
                " DOI={10.1107/S0108768104028344},"
                " journal={Acta Crystallographica Section B},"
                " author={Willighagen, E. L. and Wehrens, R. and Verwer,"
                " P. and de Gelder R. and Buydens, L. M. C.}, year={2005}}"]

    def implementors(self):
        return ["Nils E. R. Zimmermann"]


class CoulombMatrix(BaseFeaturizer):
    """
    Generate the Coulomb matrix, a representation of nuclear coulombic interaction.

    Generate the Coulomb matrix, M, of the input
    structure (or molecule).  The Coulomb matrix was put forward by
    Rupp et al. (Phys. Rev. Lett. 108, 058301, 2012) and is defined by
    off-diagonal elements M_ij = Z_i*Z_j/|R_i-R_j|
    and diagonal elements 0.5*Z_i^2.4, where Z_i and R_i denote
    the nuclear charge and the position of atom i, respectively.

    Args:
        diag_elems: (bool) flag indicating whether (True, default) to use
                    the original definition of the diagonal elements;
                    if set to False, the diagonal elements are set to zero.
    """

    def __init__(self, diag_elems=True):
        self.diag_elems = diag_elems

    def featurize(self, s):
        """
        Get Coulomb matrix of input structure.

        Args:
            s: input Structure (or Molecule) object.

        Returns:
            m: (Nsites x Nsites matrix) Coulomb matrix.
        """
        m = [[] for site in s.sites]
        z = []
        for site in s.sites:
            if isinstance(site.specie, Element):
                z.append(site.specie.Z)
            else:
                z.append(site.specie.element.Z)
        for i in range(s.num_sites):
            for j in range(s.num_sites):
                if i == j:
                    if self.diag_elems:
                        m[i].append(0.5 * z[i] ** 2.4)
                    else:
                        m[i].append(0)
                else:
                    d = s.get_distance(i, j) * ANG_TO_BOHR
                    m[i].append(z[i] * z[j] / d)
        return [np.array(m)]

    def feature_labels(self):
        return ["coulomb matrix"]

    def citations(self):
        return ["@article{rupp_tkatchenko_muller_vonlilienfeld_2012, title={"
                "Fast and accurate modeling of molecular atomization energies"
                " with machine learning}, volume={108},"
                " DOI={10.1103/PhysRevLett.108.058301}, number={5},"
                " pages={058301}, journal={Physical Review Letters}, author={"
                "Rupp, Matthias and Tkatchenko, Alexandre and M\"uller,"
                " Klaus-Robert and von Lilienfeld, O. Anatole}, year={2012}}"]

    def implementors(self):
        return ["Nils E. R. Zimmermann"]


class SineCoulombMatrix(BaseFeaturizer):
    """
    A variant of the Coulomb matrix developed for periodic crystals.

    This function generates a variant of the Coulomb matrix developed
    for periodic crystals by Faber et al. (Inter. J. Quantum Chem.
    115, 16, 2015). It is identical to the Coulomb matrix, except
    that the inverse distance function is replaced by the inverse of a
    sin**2 function of the vector between the sites which is periodic
    in the dimensions of the structure lattice. See paper for details.

    Args:
        diag_elems (bool): flag indication whether (True, default) to use
                the original definition of the diagonal elements;
                if set to False, the diagonal elements are set to 0
    """

    def __init__(self, diag_elems=True):
        self.diag_elems = diag_elems

    def featurize(self, s):
        """
        Args:
            s (Structure or Molecule): input structure (or molecule)

        Returns:
            (Nsites x Nsites matrix) Sine matrix.
        """
        sites = s.sites
        Zs = np.array([site.specie.Z for site in sites])
        sin_mat = np.zeros((len(sites), len(sites)))
        coords = np.array([site.frac_coords for site in sites])
        lattice = s.lattice.matrix
        pi = np.pi

        for i in range(len(sin_mat)):
            for j in range(len(sin_mat)):
                if i == j:
                    if self.diag_elems:
                        sin_mat[i][i] = 0.5 * Zs[i] ** 2.4
                elif i < j:
                    vec = coords[i] - coords[j]
                    coord_vec = np.sin(pi * vec) ** 2
                    trig_dist = np.linalg.norm(
                        (np.matrix(coord_vec) * lattice).A1) * ANG_TO_BOHR
                    sin_mat[i][j] = Zs[i] * Zs[j] / trig_dist
                else:
                    sin_mat[i][j] = sin_mat[j][i]
        return [sin_mat]

    def feature_labels(self):
        return ["sine coulomb matrix"]

    def citations(self):
        return ["@article {QUA:QUA24917,"
                "author = {Faber, Felix and Lindmaa, Alexander and von Lilienfeld, O. Anatole and Armiento, Rickard},"
                "title = {Crystal structure representations for machine learning models of formation energies},"
                "journal = {International Journal of Quantum Chemistry},"
                "volume = {115},"
                "number = {16},"
                "issn = {1097-461X},"
                "url = {http://dx.doi.org/10.1002/qua.24917},"
                "doi = {10.1002/qua.24917},"
                "pages = {1094--1101},"
                "keywords = {machine learning, formation energies, representations, crystal structure, periodic systems},"
                "year = {2015},"
                "}"]

    def implementors(self):
        return ["Kyle Bystrom"]


class OrbitalFieldMatrix(BaseFeaturizer):
    """
    Representation based on the valence shell electrons of neighboring atoms.

    Each atom is described by a 32-element vector (or 39-element vector, see
    period tag for details) uniquely representing the valence subshell.
    A 32x32 (39x39) matrix is formed by multiplying two atomic vectors.
    An OFM for an atomic environment is the sum of these matrices for each atom
    the center atom coordinates with multiplied by a distance function
    (In this case, 1/r times the weight of the coordinating atom in the Voronoi
     Polyhedra method). The OFM of a structure or molecule is the average of the
     OFMs for all the sites in the structure.

    Args:
        period_tag (bool): In the original OFM, an element is represented
            by a vector of length 32, where each element is 1 or 0,
            which represents the valence subshell of the element.
            With period_tag=True, the vector size is increased
            to 39, where the 7 extra elements represent the period
            of the element. Note lanthanides are treated as period 6,
            actinides as period 7. Default False as in the original paper.
        flatten (bool): Flatten the avg OFM to a 1024-vector (if period_tag
            False) or a 1521-vector (if period_tag=True).

    ...attribute:: size
        Either 32 or 39, the size of the vectors used to describe elements.

    Reference:
        `Pham et al. _Sci Tech Adv Mat_. 2017 <http://dx.doi.org/10.1080/14686996.2017.1378060>_`
    """

    def __init__(self, period_tag=False, flatten=False):
        """Initialize the featurizer

        Args:
            period_tag (bool): In the original OFM, an element is represented
                    by a vector of length 32, where each element is 1 or 0,
                    which represents the valence subshell of the element.
                    With period_tag=True, the vector size is increased
                    to 39, where the 7 extra elements represent the period
                    of the element. Note lanthanides are treated as period 6,
                    actinides as period 7. Default False as in the original paper.
        """
        my_ohvs = {}
        if period_tag:
            self.size = 39
        else:
            self.size = 32
        for Z in range(1, 95):
            el = Element.from_Z(Z)
            my_ohvs[Z] = self.get_ohv(el, period_tag)
            my_ohvs[Z] = np.matrix(my_ohvs[Z])
        self.ohvs = my_ohvs
        self.flatten = flatten

    def get_ohv(self, sp, period_tag):
        """
        Get the "one-hot-vector" for pymatgen Element sp. This 32 or 39-length
        vector represents the valence shell of the given element.
        Args:
            sp (Element): element whose ohv should be returned
            period_tag (bool): If true, the vector contains items
                    corresponding to the period of the element

        Returns:
            my_ohv (numpy array length 39 if period_tag, else 32): ohv for sp
        """
        el_struct = sp.full_electronic_structure
        ohd = {j: {i + 1: 0 for i in range(2 * (2 * j + 1))} for j in range(4)}
        nume = 0
        shell_num = 0
        max_n = el_struct[-1][0]
        while shell_num < len(el_struct):
            if el_struct[-1 - shell_num][0] < max_n - 2:
                shell_num += 1
                continue
            elif el_struct[-1 - shell_num][0] < max_n - 1 and \
                    el_struct[-1 - shell_num][1] != u'f':
                shell_num += 1
                continue
            elif el_struct[-1 - shell_num][0] < max_n and (
                    el_struct[-1 - shell_num][1] != u'd' and
                    el_struct[-1 - shell_num][1] != u'f'):
                shell_num += 1
                continue
            curr_shell = el_struct[-1 - shell_num]
            if curr_shell[1] == u's':
                l = 0
            elif curr_shell[1] == u'p':
                l = 1
            elif curr_shell[1] == u'd':
                l = 2
            elif curr_shell[1] == u'f':
                l = 3
            ohd[l][curr_shell[2]] = 1
            nume += curr_shell[2]
            shell_num += 1
        my_ohv = np.zeros(self.size, np.int)
        k = 0
        for j in range(4):
            for i in range(2 * (2 * j + 1)):
                my_ohv[k] = ohd[j][i + 1]
                k += 1
        if period_tag:
            row = sp.row
            if row > 7:
                row -= 2
            my_ohv[row + 31] = 1
        return my_ohv

    def get_single_ofm(self, site, site_dict):
        """
        Gets the orbital field matrix for a single chemical environment,
        where site is the center atom whose environment is characterized and
        site_dict is a dictionary of site : weight, where the weights are the
        Voronoi Polyhedra weights of the corresponding coordinating sites.

        Args:
            site (Site): center atom
            site_dict (dict of Site:float): chemical environment

        Returns:
            atom_ofm (size X size numpy matrix): ofm for site
        """
        ohvs = self.ohvs
        atom_ofm = np.matrix(np.zeros((self.size, self.size)))
        ref_atom = ohvs[site.specie.Z]
        for other_site in site_dict:
            scale = other_site['weight']
            other_atom = ohvs[other_site['site'].specie.Z]
            atom_ofm += other_atom.T * ref_atom * scale / site.distance(
                other_site['site']) / ANG_TO_BOHR
        return atom_ofm

    def get_atom_ofms(self, struct, symm=False):
        """
        Calls get_single_ofm for every site in struct. If symm=True,
        get_single_ofm is called for symmetrically distinct sites, and
        counts is constructed such that ofms[i] occurs counts[i] times
        in the structure

        Args:
            struct (Structure): structure for find ofms for
            symm (bool): whether to calculate ofm for only symmetrically
                    distinct sites

        Returns:
            ofms ([size X size matrix] X len(struct)): ofms for struct
            if symm:
                ofms ([size X size matrix] X number of symmetrically distinct sites):
                    ofms for struct
                counts: number of identical sites for each ofm
        """
        ofms = []
        vnn = pmg_le.VoronoiNN(allow_pathological=True)
        if symm:
            symm_struct = SpacegroupAnalyzer(struct).get_symmetrized_structure()
            indices = [lst[0] for lst in symm_struct.equivalent_indices]
            counts = [len(lst) for lst in symm_struct.equivalent_indices]
        else:
            indices = [i for i in range(len(struct.sites))]
        for index in indices:
            ofms.append(self.get_single_ofm(struct.sites[index],
                                            vnn.get_nn_info(struct, index)))
        if symm:
            return ofms, counts
        return ofms

    def get_mean_ofm(self, ofms, counts):
        """
        Averages a list of ofms, weights by counts
        """
        ofms = [ofm * c for ofm, c in zip(ofms, counts)]
        return sum(ofms) / sum(counts)

    def get_structure_ofm(self, struct):
        """
        Calls get_mean_ofm on the results of get_atom_ofms
        to give a size X size matrix characterizing a structure
        """
        ofms, counts = self.get_atom_ofms(struct, True)
        return self.get_mean_ofm(ofms, counts)

    def featurize(self, s):
        """
        Makes a supercell for structure s (to protect sites
        from coordinating with themselves), and then finds the mean
        of the orbital field matrices of each site to characterize
        a structure

        Args:
            s (Structure): structure to characterize

        Returns:
            mean_ofm (size X size matrix): orbital field matrix
                    characterizing s
        """
        s *= [3, 3, 3]
        ofms, counts = self.get_atom_ofms(s, True)
        mean_ofm = self.get_mean_ofm(ofms, counts)
        if self.flatten:
            return mean_ofm.A.flatten()
        else:
            return [mean_ofm.A]

    def feature_labels(self):
        if self.flatten:
            slabels = ["s^{}".format(i) for i in range(1, 3)]
            plabels = ["p^{}".format(i) for i in range(1, 7)]
            dlabels = ["d^{}".format(i) for i in range(1, 11)]
            flabels = ["f^{}".format(i) for i in range(1, 15)]
            labelset_1D = slabels + plabels + dlabels + flabels

            # account for period tags
            if self.size==39:
                period_labels = ["period {}".format(i) for i in range(1, 8)]
                labelset_1D += period_labels

            labelset_2D = []
            for l1 in labelset_1D:
                for l2 in labelset_1D:
                    labelset_2D.append('OFM: ' + l1 + ' - ' + l2)
            return labelset_2D
        else:
            return ["orbital field matrix"]

    def citations(self):
        return ["@article{LamPham2017,"
                "author = {{Lam Pham}, Tien and Kino, Hiori and Terakura, Kiyoyuki and "
                "Miyake, Takashi and Tsuda, Koji and Takigawa, Ichigaku and {Chi Dam}, Hieu},"
                "doi = {10.1080/14686996.2017.1378060},"
                "journal = {Science and Technology of Advanced Materials},"
                "month = {dec},"
                "number = {1},"
                "pages = {756--765},"
                "publisher = {Taylor {\&} Francis},"
                "title = {{Machine learning reveals orbital interaction in materials}},"
                "url = {https://www.tandfonline.com/doi/full/10.1080/14686996.2017.1378060},"
                "volume = {18},"
                "year = {2017}"
                "}"]

    def implementors(self):
        return ["Kyle Bystrom", "Alex Dunn"]


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
            min_rel_dists: (list of floats) list of all minimum relative
                    distances (i.e., for all sites).
        """
        vire = ValenceIonicRadiusEvaluator(s)
        min_rel_dists = []
        for site in vire.structure:
            min_rel_dists.append(min([dist / (
                    vire.radii[site.species_string] +
                    vire.radii[neigh.species_string]) for neigh, dist in \
                                      vire.structure.get_neighbors(site,
                                                                   self.cutoff)]))
        return [min_rel_dists[:]]

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
        return ["Nils E. R. Zimmermann"]


class SiteStatsFingerprint(BaseFeaturizer):
    """
    Computes statistics of properties across all sites in a structure.

    This featurizer first uses a site featurizer class (see site.py for
    options) to compute features of each site in a structure, and then computes
    features of the entire structure by measuring statistics of each attribute.
    Can optionally compute the the statistics of only sites with certain ranges
    of oxidation states (e.g., only anions).

    Features:
        - Returns each statistic of each site feature
    """

    def __init__(self, site_featurizer, stats=('mean', 'std_dev'), min_oxi=None,
                 max_oxi=None, covariance=False):
        """
        Args:
            site_featurizer (BaseFeaturizer): a site-based featurizer
            stats ([str]): list of weighted statistics to compute for each feature.
                If stats is None, a list is returned for each features
                that contains the calculated feature for each site in the
                structure.
                *Note for nth mode, stat must be 'n*_mode'; e.g. stat='2nd_mode'
            min_oxi (int): minimum site oxidation state for inclusion (e.g.,
                zero means metals/cations only)
            max_oxi (int): maximum site oxidation state for inclusion
            covariance (bool): Whether to compute the covariance of site features
        """

        self.site_featurizer = site_featurizer
        self.stats = tuple([stats]) if type(stats) == str else stats
        if self.stats and '_mode' in ''.join(self.stats):
            nmodes = 0
            for stat in self.stats:
                if '_mode' in stat and int(stat[0]) > nmodes:
                    nmodes = int(stat[0])
            self.nmodes = nmodes

        self.min_oxi = min_oxi
        self.max_oxi = max_oxi
        self.covariance = covariance

    @property
    def _site_labels(self):
        return self.site_featurizer.feature_labels()

    def featurize(self, s):
        # Get each feature for each site
        vals = [[] for t in self._site_labels]
        for i, site in enumerate(s.sites):
            if (self.min_oxi is None or site.specie.oxi_state >= self.min_oxi) \
                    and (self.max_oxi is None or site.specie.oxi_state >= self.max_oxi):
                opvalstmp = self.site_featurizer.featurize(s, i)
                for j, opval in enumerate(opvalstmp):
                    if opval is None:
                        vals[j].append(0.0)
                    else:
                        vals[j].append(opval)

        # If the user does not request statistics, return the site features now
        if self.stats is None:
            return vals

        # Compute the requested statistics
        stats = []
        for op in vals:
            for stat in self.stats:
                stats.append(PropertyStats().calc_stat(op, stat))

        # If desired, compute covariances
        if self.covariance:
            if len(s) == 1:
                stats.extend([0] * int(len(vals) * (len(vals) - 1) / 2))
            else:
                covar = np.cov(vals)
                tri_ind = np.triu_indices(len(vals), 1)
                stats.extend(covar[tri_ind].tolist())

        return stats

    def feature_labels(self):
        if self.stats:
            labels = []
            # Make labels associated with the statistics
            for attr in self._site_labels:
                for stat in self.stats:
                    labels.append('%s %s' % (stat, attr))

            # Make labels associated with the site labels
            if self.covariance:
                sl = self._site_labels
                for i, sa in enumerate(sl):
                    for sb in sl[(i+1):]:
                        labels.append('covariance %s-%s'%(sa, sb))
            return labels
        else:
            return self._site_labels

    def citations(self):
        return self.site_featurizer.citations()

    def implementors(self):
        return ['Nils E. R. Zimmermann', 'Alireza Faghaninia',
                'Anubhav Jain', 'Logan Ward']

    @staticmethod
    def from_preset(preset, **kwargs):
        """
        Create a SiteStatsFingerprint class according to a preset

        Args:
            preset (str) - Name of preset
            kwargs - Options for SiteStatsFingerprint
        """

        if preset == "CrystalNNFingerprint_cn":
            return SiteStatsFingerprint(
                CrystalNNFingerprint.from_preset("cn", cation_anion=False),
                **kwargs)

        elif preset == "CrystalNNFingerprint_cn_cation_anion":
            return SiteStatsFingerprint(
                CrystalNNFingerprint.from_preset("cn", cation_anion=True),
                **kwargs)

        elif preset == "CrystalNNFingerprint_ops":
            return SiteStatsFingerprint(
                CrystalNNFingerprint.from_preset("ops", cation_anion=False),
                **kwargs)

        elif preset == "CrystalNNFingerprint_ops_cation_anion":
            return SiteStatsFingerprint(
                CrystalNNFingerprint.from_preset("ops", cation_anion=True),
                **kwargs)

        elif preset == "OPSiteFingerprint":
            return SiteStatsFingerprint(OPSiteFingerprint(), **kwargs)

        elif preset == "OPSiteFingerprint":
            return SiteStatsFingerprint(OPSiteFingerprint(), **kwargs)

        elif preset == "LocalPropertyDifference_ward-prb-2017":
            return SiteStatsFingerprint(
                LocalPropertyDifference.from_preset("ward-prb-2017"),
                stats=["minimum", "maximum", "range", "mean", "avg_dev"]
            )

        elif preset == "CoordinationNumber_ward-prb-2017":
            return SiteStatsFingerprint(
                CoordinationNumber(nn=VoronoiNN(weight='area'),
                                   use_weights="effective"),
                stats=["minimum", "maximum", "range", "mean", "avg_dev"]
            )

        elif preset == "Composition-dejong2016_AD":
            return SiteStatsFingerprint(LocalPropertyDifference(
                properties=["Number", "AtomicWeight",
                            "Column", "Row", "CovalentRadius",
                            "Electronegativity"], signed=False),
                stats=['holder_mean::%d' % d for d in range(0, 4 + 1)] + ['std_dev'],
            )

        elif preset == "Composition-dejong2016_SD":
            return SiteStatsFingerprint(LocalPropertyDifference(
                properties=["Number", "AtomicWeight",
                            "Column", "Row", "CovalentRadius",
                            "Electronegativity"], signed=True),
                stats=['holder_mean::%d' % d for d in [1, 2, 4]] + ['std_dev'],
            )

        elif preset == "BondLength-dejong2016":
            return SiteStatsFingerprint(AverageBondLength(VoronoiNN()),
                                        stats=['holder_mean::%d' % d for d in range(-4, 4 + 1)]
                                              + ['std_dev', 'geom_std_dev'])

        elif preset == "BondAngle-dejong2016":
            return SiteStatsFingerprint(AverageBondAngle(VoronoiNN()),
                                        stats=['holder_mean::%d' % d for d in range(-4, 4 + 1)]
                                              + ['std_dev', 'geom_std_dev'])

        else:
            # TODO: Why assume coordination number? Should this just raise an error? - lw
            # One of the various Coordination Number presets:
            # MinimumVIRENN, MinimumDistanceNN, JMolNN, VoronoiNN, etc.
            try:
                return SiteStatsFingerprint(
                    CoordinationNumber.from_preset(preset), **kwargs)
            except:
                pass

        raise ValueError("Unrecognized preset!")


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


class BondFractions(BaseFeaturizer):
    """
    Compute the fraction of each bond in a structure, based on NearestNeighbors.

    For example, in a structure with 2 Li-O bonds and 3 Li-P bonds:

    Li-0: 0.4
    Li-P: 0.6

    Features:

    BondFractions must be fit with iterable of structures before featurization in
    order to define the allowed bond types (features). To do this, pass a list
    of allowed_bonds. Otherwise, fit based on a list of structures. If
    allowed_bonds is defined and BondFractions is also fit, the intersection
    of the two lists of possible bonds is used.

    For dataframes containing structures of various compositions, a unified
    dataframe is returned which has the collection of all possible bond types
    gathered from all structures as columns. To approximate bonds based on
    chemical rules (ie, for a structure which you'd like to featurize but has
    bonds not in the allowed set), use approx_bonds = True.

    BondFractions is based on the "sum over bonds" in the Bag of Bonds approach,
    based on a method by Hansen et. al "Machine Learning Predictions of Molecular
    Properties: Accurate Many-Body Potentials and Nonlocality in Chemical Space"
    (2015).

    Args:
        nn (NearestNeighbors): A Pymatgen nearest neighbors derived object. For
            example, pymatgen.analysis.local_env.VoronoiNN().
        bbv (float): The 'bad bond values', values substituted for
            structure-bond combinations which can not physically exist, but
            exist in the unified dataframe. For example, if a dataframe contains
            structures of BaLiP and BaTiO3, determines the value to place in
            the Li-P column for the BaTiO3 row; by default, is 0.
        no_oxi (bool): If True, the featurizer will be agnostic to oxidation
            states, which prevents oxidation states from  differentiating
            bonds. For example, if True, Ca - O is identical to Ca2+ - O2-,
            Ca3+ - O-, etc., and all of them will be included in Ca - O column.
        approx_bonds (bool): If True, approximates the fractions of bonds not
            in allowed_bonds (forbidden bonds) with similar allowed bonds.
            Chemical rules are used to determine which bonds are most 'similar';
            particularly, the Euclidean distance between the 2-tuples of the
            bonds in Mendeleev no. space is minimized for the approximate
            bond chosen.
        token (str): The string used to separate species in a bond, including
            spaces. The token must contain at least one space and cannot have
            alphabetic characters in it, and should be padded by spaces. For
            example, for the bond Cs+ - Cl-, the token is ' - '. This determines
            how bonds are represented in the dataframe.
        allowed_bonds ([str]): A listlike object containing bond types as
            strings. For example, Cs - Cl, or Li+ - O2-. Ions and elements
            will still have distinct bonds if (1) the bonds list originally
            contained them and (2) no_oxi is False. These must match the
            token specified.
    """

    def __init__(self, nn=pmg_le.CrystalNN(), bbv=0, no_oxi=False,
                 approx_bonds=False, token=' - ', allowed_bonds=None):
        self.nn = nn
        self.bbv = bbv
        self.no_oxi = no_oxi
        self.approx_bonds = approx_bonds

        if " " not in token:
            raise ValueError("A space must be present in the token.")

        if any([str.isalnum(i) for i in token]):
            raise ValueError("The token cannot have any alphanumeric "
                             "characters.")

        token_els = token.split(" ")
        if len(token_els) != 3 and token != " ":
            raise ValueError("The token must either be a space or be padded by"
                             "single spaces with no spaces in between.")

        self.token = token

        if allowed_bonds is None:
            self.allowed_bonds = allowed_bonds
            self.fitted_bonds_ = allowed_bonds
        else:
            self.allowed_bonds = self._sanitize_bonds(allowed_bonds)
            self.fitted_bonds_ = self._sanitize_bonds(allowed_bonds)


    @staticmethod
    def from_preset(preset, **kwargs):
        """
        Use one of the standard instances of a given NearNeighbor class.
        Pass args to __init__, such as allowed_bonds, using this method as well.

        Args:
            preset (str): preset type ("CrystalNN", "VoronoiNN", "JMolNN",
            "MiniumDistanceNN", "MinimumOKeeffeNN", or "MinimumVIRENN").

        Returns:
            CoordinationNumber from a preset.
        """
        nn = getattr(pmg_le, preset)
        return BondFractions(nn(), **kwargs)

    def fit(self, X, y=None):
        """
        Define the bond types allowed to be returned during each featurization.
        Bonds found during featurization which are not allowed will be omitted
        from the returned dataframe or matrix.

        Fit BondFractions by either passing an iterable of structures to
        training_data or by defining the bonds explicitly with allowed_bonds
        in __init__.

        Args:
            X (Series/list): An iterable of pymatgen Structure
                objects which will be used to determine the allowed bond
                types.
            y : unused (added for consistency with overridden method signature)

        Returns:
            self

        """
        if not hasattr(X, "__getitem__"):
            raise ValueError("X must be an iterable of pymatgen Structures")

        X = X.values if isinstance(X, pd.Series) else X

        if not all([isinstance(x, Structure) for x in X]):
            raise ValueError("Each structure must be a pymatgen Structure "
                             "object.")

        sanitized = self._sanitize_bonds(self.enumerate_all_bonds(X))

        if self.allowed_bonds is None:
            self.fitted_bonds_ = sanitized
        else:
            self.fitted_bonds_ = [b for b in sanitized if b in self.allowed_bonds]
            if len(self.fitted_bonds_) == 0:
                warnings.warn("The intersection between the allowed bonds "
                              "and the fitted bonds is zero. There's no bonds"
                              "to be featurized!")

        return self

    def enumerate_bonds(self, s):
        """
        Lists out all the bond possibilities in a single structure.

        Args:
            s (Structure): A pymatgen structure

        Returns:
            A list of bond types in 'Li-O' form, where the order of the
            elements in each bond type is alphabetic.
        """
        els = s.composition.elements
        het_bonds = list(itertools.combinations(els, 2))
        het_bonds = [tuple(sorted([str(i) for i in j])) for j in het_bonds]
        hom_bonds = [(str(el), str(el)) for el in els]
        bond_types = [k[0] + self.token + k[1] for k in het_bonds + hom_bonds]
        return sorted(bond_types)

    def enumerate_all_bonds(self, structures):
        """
        Identify all the unique, possible bonds types of all structures present,
        and create the 'unified' bonds list.

        Args:
             structures (list/ndarray): List of pymatgen Structures

        Returns:
            A tuple of unique, possible bond types for an entire list of
            structures. This tuple is used to form the unified feature labels.
        """
        bond_types = []
        for s in structures:
            bts = self.enumerate_bonds(s)
            for bt in bts:
                if bt not in bond_types:
                    bond_types.append(bt)
        return tuple(sorted(bond_types))

    def featurize(self, s):
        """
        Quantify the fractions of each bond type in a structure.

        For collections of structures, bonds types which are not found in a
        particular structure (e.g., Li-P in BaTiO3) are represented as NaN.

        Args:
            s (Structure): A pymatgen Structure object

        Returns:
            (list) The feature list of bond fractions, in the order of the
                alphabetized corresponding bond names.
        """

        self._check_fitted()

        bond_types = tuple(self.enumerate_bonds(s))
        bond_types = self._sanitize_bonds(bond_types)
        bonds = {k: 0.0 for k in bond_types}
        tot_bonds = 0.0

        # if we find a bond in allowed_bonds not in bond_types, mark as bbv
        for b in self.fitted_bonds_:
            if b not in bond_types:
                if self.bbv is None:
                    bonds[b] = float("nan")
                else:
                    bonds[b] = self.bbv

        for i, _ in enumerate(s.sites):
            nearest = self.nn.get_nn(s, i)
            origin = s.sites[i].specie

            for neigh in nearest:
                btup = tuple(sorted([str(origin), str(neigh.specie)]))
                b = self._sanitize_bonds(btup[0] + self.token + btup[1])
                # The bond will not be in bonds if it is a forbidden bond
                # (when a local bond is not in allowed_bonds)
                tot_bonds += 1.0
                if b in bonds:
                    bonds[b] += 1.0

        if self.approx_bonds:
            bonds = self._approximate_bonds(bonds)

        # If allowed_bonds caused no bonds to be present, all bonds will be 0.
        # Prevent division by zero error.
        tot_bonds = tot_bonds or 1.0

        # if we find a bond in bond_types not in allowed_bonds, skip
        return [bonds[b] / tot_bonds for b in self.fitted_bonds_]

    def feature_labels(self):
        """
        Returns the list of allowed bonds. Throws an error if the featurizer
        has not been fit.
        """
        self._check_fitted()
        return [b + " bond frac." for b in self.fitted_bonds_]

    def _check_fitted(self):
        """
        Ensure the Featurizer has been fit to the dataframe
        """
        if self.fitted_bonds_ is None:
            raise NotFittedError('BondFractions must have a list of allowed bonds.'
                                 ' Either pass in a list of bonds to the '
                                 'initializer with allowed_bonds, use "fit" with'
                                 ' a list of structures, or do both to sets the '
                                 'intersection of the two as the allowed list.')

    def _sanitize_bonds(self, bonds):
        """
        Prevent errors and/or bond duplicates from badly formatted allowed_bonds

        Args:
            bonds (str/[str]): An iterable of bond types, specified as strings
                with the general format "El - Sp", where El or Sp can be specie
                or an element with pymatgen's str representation of a bond. For
                example, a Cesium Chloride bond could be represented as either
                "Cs-Cl" or "Cs+-Cl-" or "Cl-Cs" or "Cl--Cs+". "bond frac." may
                be present at the end of each bond, as it will be sanitized.
                Can also be a single string bond type.
        Returns:
            bonds ([str]): A listlike object containing alphabetized bond types.
                Note that ions and elements will still have distinct bonds if
                the bonds list originally contained them.
        """
        if isinstance(bonds, str):
            single = True
            bonds = [bonds]
        else:
            single = False
            try:
                bonds = list(bonds)
            except:
                # In the case of a series object
                bonds = bonds.tolist()

        for i, bond in enumerate(bonds):
            if not isinstance(bond, str):
                raise TypeError("Bonds must be specified as strings between "
                                "elements or species with the token in between, "
                                "for example Cl - Cs")
            if not self.token in bond:
                raise ValueError('Token "{}" not found in bond: {}'.format(
                    self.token, bond))
            bond = bond.replace(" bond frac.", "")
            species = sorted(bond.split(self.token))

            if self.no_oxi:
                alphabetized = self.token.join(species)
                species = self._species_from_bondstr(alphabetized)
                species = [str(s.element) for s in species]

            bonds[i] = self.token.join(species)
        bonds = list(OrderedDict.fromkeys(bonds))

        if single:
            return bonds[0]
        else:
            return tuple(sorted(bonds))

    def _species_from_bondstr(self, bondstr):
        """
        Create a 2-tuple of species objects from a bond string.

        Args:
            bondstr (str): A string representing a bond between elements or
                species, or a combination of the two. For example, "Cl- - Cs+".

        Returns:
            ((Species)): A tuple of pymatgen Species objects in alphabetical
                order.
        """
        species = []
        for ss in bondstr.split(self.token):
            try:
                species.append(Specie.from_string(ss))
            except ValueError:
                d = {'element': ss, 'oxidation_state': 0}
                species.append(Specie.from_dict(d))
        return tuple(species)

    def _approximate_bonds(self, local_bonds):
        """
        Approximate a structure's bonds if the structure contains bonds not in
        allowed_bonds.

        Local bonds are approximated according to the "nearest" bonds present in
        allowed_bonds (the unified list). Nearness is measured by the euclidean
        distance (diff) in mendeleev number of each element. For example a Na-O
        bond could be approximated as a Li-O bond ( distance is sqrt(0^2 + 1^2)
         = 1).

        Args:
            local_bonds (dict): The bonds present in the structure with the bond
                types as keys ("Cl- - Cs+") and the bond fraction as values
                (0.7).

        Returns:
            abonds_data (dict): A dictionary of the unified (allowed) bonds
                with the bond names as keys and the corresponding bond fractions
                (whether approximated or true) as values.

        """

        # At this stage, local_bonds may contain unified bonds which
        # are nan.

        abonds_data = {k: 0.0 for k in self.fitted_bonds_}
        abonds_species = {k: None for k in self.fitted_bonds_}
        for ub in self.fitted_bonds_:
            species = self._species_from_bondstr(ub)
            abonds_species[ub] = tuple(species)
        # keys are pairs of species, values are bond names in unified_bonds
        abonds_species = {v: k for k, v in abonds_species.items()}

        for lb in local_bonds.keys():
            local_bonds[lb] = 0.0 if np.isnan(local_bonds[lb]) else local_bonds[lb]

            if lb in self.fitted_bonds_:
                abonds_data[lb] += local_bonds[lb]
            else:
                lbs = self._species_from_bondstr(lb)

                nearest = []
                d_min = None
                for abss in abonds_species.keys():

                    # The distance between bonds is euclidean. To get a good
                    # measure of the coordinate between mendeleev numbers for
                    # each specie, we use the minumum difference. ie, for
                    # finding the distance between Na-O and O-Li, we would
                    # not want the distance between (Na and O) and (O and Li),
                    # we want the distance between (Na and Li) and (O and O).

                    u_mends = sorted([j.element.mendeleev_no for j in abss])
                    l_mends = sorted([j.element.mendeleev_no for j in lbs])

                    d0 = u_mends[0] - l_mends[0]
                    d1 = u_mends[1] - l_mends[1]

                    d = (d0**2.0 + d1**2.0)**0.5
                    if not d_min:
                        d_min = d
                        nearest = [abss]
                    elif d < d_min:
                        # A new best approximation has been found
                        d_min = d
                        nearest = [abss]
                    elif d == d_min:
                        # An equivalent approximation has been found
                        nearest += [abss]
                    else:
                        pass

                # Divide bond fraction equally among all equiv. approximate bonds
                bond_frac = local_bonds[lb]/len(nearest)
                for n in nearest:
                    # Get the name of the approximate bond from the map
                    ab = abonds_species[n]

                    # Add the bond frac to that/those nearest bond(s)
                    abonds_data[ab] += bond_frac
        return abonds_data

    def implementors(self):
        return ["Alex Dunn"]

    def citations(self):
        return ["@article{doi:10.1021/acs.jpclett.5b00831, "
                "author = {Hansen, Katja and Biegler, "
                "Franziska and Ramakrishnan, Raghunathan and Pronobis, Wiktor"
                "and von Lilienfeld, O. Anatole and Muller, Klaus-Robert and"
                "Tkatchenko, Alexandre},"
                "title = {Machine Learning Predictions of Molecular Properties: "
                "Accurate Many-Body Potentials and Nonlocality in Chemical Space},"
                "journal = {The Journal of Physical Chemistry Letters},"
                "volume = {6},"
                "number = {12},"
                "pages = {2326-2331},"
                "year = {2015},"
                "doi = {10.1021/acs.jpclett.5b00831}, "
                "note ={PMID: 26113956},"
                "URL = {http://dx.doi.org/10.1021/acs.jpclett.5b00831}"
                "}"]


class BagofBonds(BaseFeaturizer):
    """
    Compute a Bag of Bonds vector, as first described by Hansen et al. (2015).

    The Bag of Bonds approach is based creating an even-length vector from a
    Coulomb matrix output. Practically, it represents the Coloumbic interactions
    between each possible set of sites in a structure as a vector.

    BagofBonds must be fit to an iterable of structures using the "fit" method
    before featurization can occur. This is because the bags and the maximum
    lengths of each bag must be set prior to featurization. We recommend
    fitting and featurizing on the same data to maintain consistency
    between generated feature sets. This can be done using the fit_transform
    method (for lists of structures) or the fit_featurize_dataframe method
    (for dataframes).

    BagofBonds is based on a method by Hansen et. al "Machine Learning
    Predictions of Molecular Properties: Accurate Many-Body Potentials and
    Nonlocality in Chemical Space" (2015).

    Args:
        coulomb_matrix (BaseFeaturizer): A featurizer object containing a
            "featurize" method which returns a matrix of size nsites x nsites.
            Good choices are CoulombMatrix() or SineCoulombMatrix()
        token (str): The string used to separate species in a bond, including
            spaces. The token must contain at least one space and cannot have
            alphabetic characters in it, and should be padded by spaces. For
            example, for the bond Cs+ - Cl-, the token is ' - '. This determines
            how bonds are represented in the dataframe.

    """
    def __init__(self, coulomb_matrix=CoulombMatrix(), token=' - '):
        self.coulomb_matrix = coulomb_matrix
        self.token = token

    def _check_fitted(self):
        if not hasattr(self, 'baglens') or not hasattr(self, 'ordered_bonds'):
            raise NotFittedError("BagofBonds not fitted to any list of "
                                 "structures! Use the 'fit' method to define "
                                 "the bags and the maximum length of each bag.")

    def fit(self, X, y=None):
        """
        Define the bags using a list of structures.

        Both the names of the bags (e.g., Cs-Cl) and the maximum lengths of
        the bags are set with fit.

        Args:
            X (Series/list): An iterable of pymatgen Structure
                objects which will be used to determine the allowed bond
                types and bag lengths.
            y : unused (added for consistency with overridden method signature)

        Returns:
            self

        """
        unpadded_bobs = [self.bag(s, return_baglens=True) for s in X]
        bonds = [list(bob.keys()) for bob in unpadded_bobs]
        bonds = np.unique(sum(bonds, []))
        baglens = [0]*len(bonds)

        for i, bond in enumerate(bonds):
            for bob in unpadded_bobs:
                if bond in bob:
                    baglen = bob[bond]
                    baglens[i] = max((baglens[i], baglen))
        self.baglens = dict(zip(bonds, baglens))
        # Sort the bags by bag length, with the shortest coming first.
        self.ordered_bonds = [b[0] for b in sorted(self.baglens.items(),
                                    key=lambda bl: bl[1])]
        return self

    def bag(self, s, return_baglens=False):
        """
        Convert a structure into a bag of bonds, where each bag has no padded
        zeros. using this function will give the 'raw' bags, which when
        concatenated, will have different lengths.

        Args:
            s (Structure): A pymatgen Structure or IStructure object.
            return_baglens (bool): If True, returns the bag of bonds with as
                a dictionary with the number of bonds as values in place
                of the vectors of coulomb matrix vals. If False, calculates
                Coulomb matrix values and returns 'raw' bags.

        Returns:
            (dict) A bag of bonds, where the keys are sorted tuples of pymatgen
                Site objects representing bonds or sites, and the values are the
                Coulomb matrix values for that bag.
        """
        if not isinstance(s, (Structure)):
            raise TypeError("BagofBonds must take in a pymatgen Structure "
                            "object, not {}".format(type(s)))

        sites = s.sites
        nsites = len(sites)
        bonds = np.zeros((nsites, nsites), dtype=object)
        for i, si in enumerate(sites):
            for j, sj in enumerate(sites):
                el0, el1 = si.specie, sj.specie
                if isinstance(el0, Specie):
                    el0 = el0.element
                if isinstance(el1, Specie):
                    el1 = el1.element
                if i == j:
                    bonds[i, j] = (el0,)
                else:
                    bonds[i, j] = tuple(sorted((el0, el1)))

        if return_baglens:
            bags = {b: 0 for b in np.unique(bonds)}
        else:
            cm = self.coulomb_matrix.featurize(s)[0]
            bags = {b: [] for b in np.unique(bonds)}

        for i in range(nsites):
            for j in range(nsites):
                bond = bonds[i, j]
                if return_baglens:
                    # Only return length of bag
                    bags[bond] = bags[bond] + 1
                else:
                    # Calculate bond "strength"
                    cmval = cm[i, j]
                    bags[bond].append(cmval)

        if return_baglens:
            return bags
        else:
            # We must sort the magnitude of bonds in each bag
            return {bond: sorted(bags[bond]) for bond in bags}

    def featurize(self, s):
        """
        Featurizes a structure according to the bag of bonds method.
        Specifically, each structure is first bagged by flattening the
        Coulomb matrix for the structure. Then, it is zero-padded according to
        the maximum number of bonds in each bag, for the set of bags that
        BagofBonds was fit with.

        Args:
            s (Structure): A pymatgen structure object

        Returns:
            (list): The Bag of Bonds vector for the input structure
        """
        self._check_fitted()
        unpadded_bob = self.bag(s)
        padded_bob = {bag: [0.0] * int(length) for bag, length in
                      self.baglens.items()}

        for bond in unpadded_bob:
            if bond not in list(self.baglens.keys()):
                raise ValueError("{} is not in the fitted "
                                 "bonds/sites!".format(bond))
            baglen_s = len(unpadded_bob[bond])
            baglen_fit = self.baglens[bond]

            if baglen_s > baglen_fit:
                raise ValueError("The bond {} has more entries than was "
                                 "fitted for (i.e., there are more {} bonds"
                                 " in structure {} ({}) than the fitted set"
                                 " allows ({}).".format(bond, bond, s, baglen_s,
                                                        baglen_fit))
            elif baglen_s < baglen_fit:
                padded_bob[bond] = unpadded_bob[bond] + \
                                   [0.0]*(baglen_fit - baglen_s)
            else:
                padded_bob[bond] = unpadded_bob[bond]

        # Ensure the bonds are printed in correct order
        bob = [padded_bob[bond] for bond in self.ordered_bonds]
        return list(sum(bob, []))

    def feature_labels(self):
        self._check_fitted()
        labels = []
        for bag in self.ordered_bonds:
            if len(bag) == 1:
                basename = str(bag[0]) + " site #"
            else:
                basename = str(bag[0]) + self.token + str(bag[1]) + " bond #"
            bls = [basename + str(i) for i in range(self.baglens[bag])]
            labels += bls
        return labels

    def implementors(self):
        return ["Alex Dunn"]

    def citations(self):
        return ["@article{doi:10.1021/acs.jpclett.5b00831, "
                "author = {Hansen, Katja and Biegler, "
                "Franziska and Ramakrishnan, Raghunathan and Pronobis, Wiktor"
                "and von Lilienfeld, O. Anatole and Muller, Klaus-Robert and"
                "Tkatchenko, Alexandre},"
                "title = {Machine Learning Predictions of Molecular Properties: "
                "Accurate Many-Body Potentials and Nonlocality in Chemical Space},"
                "journal = {The Journal of Physical Chemistry Letters},"
                "volume = {6},"
                "number = {12},"
                "pages = {2326-2331},"
                "year = {2015},"
                "doi = {10.1021/acs.jpclett.5b00831}, "
                "note ={PMID: 26113956},"
                "URL = {http://dx.doi.org/10.1021/acs.jpclett.5b00831}"
                "}"]


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
         `Ward et al. _PRB_ 2017 <http://link.aps.org/doi/10.1103/PhysRevB.96.014107>`_
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
        for i,nn in enumerate(nns):
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
        features.append(PropertyStats.avg_dev(cell_volumes) / np.mean(cell_volumes))

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
                "url = {http://link.aps.org/doi/10.1103/PhysRevB.96.014107},"
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
                "url = {http://link.aps.org/doi/10.1103/PhysRevB.96.014107},"
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

    :math:`\alpha (t,s) = 1 - \frac{\sum_n w_n \delta (t - t_n)}{x_t \sum_n w_n}`

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
         `Ward et al. _PRB_ 2017 <http://link.aps.org/doi/10.1103/PhysRevB.96.014107>`_"""

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
            return [0]*len(self.shells)

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
        return ["mean ordering parameter shell {}".format(n) for n in self.shells]

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
                "url = {http://link.aps.org/doi/10.1103/PhysRevB.96.014107},"
                "volume = {96},year = {2017}}"]

    def implementors(self):
        return ['Logan Ward']


class StructureComposition(BaseFeaturizer):
    """
    Features related to the composition of a structure

    This class is just a wrapper that calls a composition-based featurizer
    on the composition of a Structure

    Features:
        - Depends on the featurizer
    """

    def __init__(self, featurizer=None):
        """Initialize the featurizer

        Args:
            featurizer (BaseFeaturizer) - Composition-based featurizer
        """
        self.featurizer = featurizer

    def fit(self, X, y=None, **fit_kwargs):
        # Get the compositions of each of the structures
        comps = [x.composition for x in X]

        return self.featurizer.fit(comps, y, **fit_kwargs)

    def featurize(self, strc):
        return self.featurizer.featurize(strc.composition)

    def feature_labels(self):
        return self.featurizer.feature_labels()

    def citations(self):
        return self.featurizer.citations()

    def implementors(self):
        # Written by Logan Ward, but let's just pass through the
        #  composition implementors
        return self.featurizer.implementors()


class XRDPowderPattern(BaseFeaturizer):
    """
    1D array representing powder diffraction of a structure as calculated by
    pymatgen. The powder is smeared / normalized according to gaussian_kde.
    """

    def __init__(self, two_theta_range=(0, 127), bw_method=0.05,
                 pattern_length=None, **kwargs):
        """
        Initialize the featurizer.

        Args:
            two_theta_range ([float of length 2]): Tuple for range of
                two_thetas to calculate in degrees. Defaults to (0, 90). Set to
                None if you want all diffracted beams within the limiting
                sphere of radius 2 / wavelength.
            bw_method (float): how much to smear the XRD pattern
            pattern_length (float): length of final array; defaults to one value
             per degree (i.e. two_theta_range + 1)
            **kwargs: any other arguments to pass into pymatgen's XRDCalculator,
                such as the type of radiation.
        """
        self.two_theta_range = two_theta_range
        self.bw_method = bw_method
        self.pattern_length = pattern_length or two_theta_range[1] - \
                              two_theta_range[0] + 1
        self.xrd_calc = XRDCalculator(**kwargs)

    def featurize(self, strc):
        pattern = self.xrd_calc.get_pattern(
            strc, two_theta_range=self.two_theta_range)
        x, y = pattern.x, pattern.y
        hist = []
        for x1, y1 in zip(x, y):
            num = int(y1)
            hist += [x1] * num

        kernel = gaussian_kde(hist, bw_method=self.bw_method)
        x = np.linspace(self.two_theta_range[0], self.two_theta_range[1],
                        self.pattern_length)
        y = kernel(x)

        return y

    def feature_labels(self):
        return ['xrd_{}'.format(x) for x in range(self.pattern_length)]

    def citations(self):
        return ["@article{Ong2013, author = {Ong, Shyue Ping and Richards, "
                "William Davidson and Jain, Anubhav and Hautier, "
                "Geoffroy and Kocher, Michael and Cholia, Shreyas and Gunter, "
                "Dan and Chevrier, Vincent L. and Persson, "
                "Kristin A. and Ceder, Gerbrand}, "
                "doi = {10.1016/j.commatsci.2012.10.028}, issn = {09270256}, "
                "journal = {Computational Materials Science}, month = {feb}, "
                "pages = {314--319}, "
                "publisher = {Elsevier B.V.}, title = {{Python Materials "
                "Genomics (pymatgen): A robust, open-source python "
                "library for materials analysis}}, url = "
                "{http://linkinghub.elsevier.com/retrieve/pii/S0927025612006295}, "
                "volume = {68}, year = {2013} } "]

    def implementors(self):
        return ['Anubhav Jain', 'Matthew Horton']
