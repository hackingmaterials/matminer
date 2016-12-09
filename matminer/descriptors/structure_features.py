from __future__ import division, unicode_literals
import numpy as np
import abc
import math
from pymatgen import MPRester
from pymatgen.analysis.defects import ValenceIonicRadiusEvaluator
from pymatgen.analysis.structure_analyzer import VoronoiCoordFinder
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

__authors__ = 'Anubhav Jain <ajain@lbl.gov>, Saurabh Bajaj <sbajaj@lbl.gov>, ' \
              'Nils E.R. Zimmerman <nils.e.r.zimmermann@gmail.com>'


def get_packing_fraction(s):
    if not s.is_ordered:
        raise ValueError("Disordered structure support not built yet")
    total_rad = 0
    for site in s:
        total_rad += math.pi * site.specie.atomic_radius ** 2

    return total_rad / s.volume


def get_vol_per_site(s):
    if not s.is_ordered:
        raise ValueError("Disordered structure support not built yet")

    return s.volume / len(s)


def density(s):
    return s.density


def get_rdf(structure, cutoff=20.0, bin_size=0.1):
    """
    Calculate rdf fingerprint of a given structure

    Args:
        structure: pymatgen structure object
        cutoff: (int/float) distance to calculate rdf up to
        bin_size: (int/float) size of bin to obtain rdf for

    Returns: (dict) rdf in dict format where keys indicate bin distance and values are calculated rdf for that bin.

    """
    dist_rdf = {}
    for site in structure:
        neighbors_lst = structure.get_neighbors(site, cutoff)
        for neighbor in neighbors_lst:
            rij = neighbor[1]
            bin_dist = int(rij / bin_size) * bin_size
            if bin_dist in dist_rdf:
                dist_rdf[bin_dist] += 1
            else:
                dist_rdf[bin_dist] = 1
    for bin_idx in dist_rdf:
        dist_rdf[bin_idx] /= structure.density * 4 * math.pi * (bin_idx ** 2) * bin_size
    return dist_rdf


def get_rdf_peaks(dist_rdf):
    """
    Get location of highest and second highest peaks in rdf of a structure.

    Args:
        dist_rdf: (dict) as output by the function "get_rdf", keys correspond to distances and values correspond to rdf.

    Returns: (tuple) of distances highest and second highest peaks.

    """
    distances = dist_rdf.keys()
    sorted_rdfs = sorted(dist_rdf.values(), reverse=True)
    max_rdf, second_highest_rdf = sorted_rdfs[0], sorted_rdfs[1]
    max_idx = dist_rdf.values().index(max_rdf)
    second_highest_idx = dist_rdf.values().index(second_highest_rdf)
    return distances[max_idx], distances[second_highest_idx]


def get_redf(struct, cutoff=None, dr=0.05):
    """
    This function permits the calculation of the crystal structure-inherent electronic radial distribution function
    (ReDF) according to Willighagen et al., Acta Cryst., 2005, B61, 29-36. The ReDF is a structure-integral RDF (i.e.,
    summed over all sites) in which the positions of neighboring sites are weighted by electrostatic interactions
    inferred from atomic partial charges. Atomic charges are obtained from the ValenceIonicRadiusEvaluator class.

    Args:
        struct (Structure): input Structure object.
        cutoff (float): distance up to which the ReDF is to be
                calculated (default: longest diagaonal in primitive cell)
        dr (float): width of bins ("x"-axis) of ReDF (default: 0.05 A).

    Returns: (dict) a copy of the electronic radial distribution functions (ReDF) as a dictionary. The distance list
        ("x"-axis values of ReDF) can be accessed via key 'distances'; the ReDF itself via key 'redf'.
    """
    if dr <= 0:
        raise ValueError("width of bins for ReDF must be >0")

    # make primitive
    struct = SpacegroupAnalyzer(struct).find_primitive() or struct

    # add oxidation states
    struct = ValenceIonicRadiusEvaluator(struct).structure

    if cutoff is None:
        # set cutoff to longest diagonal
        a = struct.lattice.matrix[0]
        b = struct.lattice.matrix[1]
        c = struct.lattice.matrix[2]
        cutoff = max([np.linalg.norm(a + b + c), np.linalg.norm(-a + b + c), np.linalg.norm(a - b + c),
                      np.linalg.norm(a + b - c)])

    nbins = int(cutoff / dr) + 1
    redf_dict = {"distances": np.array([(i + 0.5) * dr for i in range(nbins)]),
                 "redf": np.zeros(nbins, dtype=np.float)}

    for site in struct.sites:
        this_charge = float(site.specie.oxi_state)
        neighs_dists = struct.get_neighbors(site, cutoff)
        for neigh, dist in neighs_dists:
            neigh_charge = float(neigh.specie.oxi_state)
            bin_index = int(dist / dr)
            redf_dict["redf"][bin_index] += (this_charge * neigh_charge) / (struct.num_sites * dist)

    return redf_dict


class CNBase:
    __metaclass__ = abc.ABCMeta
    """
    This is an abstract base class for implementation of CN algorithms. All CN methods
    must be subclassed from this class, and have a compute method that returns CNs as
    a dict.
    """

    def __init__(self, params=None):
        """
        :param params: (dict) of parameters to pass to compute method.
        """
        self._params = params if params else {}
        self._cns = {}

    @abc.abstractmethod
    def compute(self, structure, n):
        """
        :param structure: (Structure) a pymatgen Structure
        :param n: (int) index of the atom in structure that the CN will be calculated
            for.
        :return: Dict of CN's for the site n. (e.g. {'O': 4.4, 'F': 2.1})
        """
        pass


class OKeefes(CNBase):
    """
    O'Keefe's CN's as implemented in pymatgen.
    Note: We only need to define the compute method using the CNBase class
        that returns a dict object.
        params can be accessed via ._params and passed to actual function.
    """
    def compute(self, structure, n):
        params = self._params
        vor = VoronoiCoordFinder(structure, **params)
        vorp = vor.get_voronoi_polyhedra(n)
        cdict = {}
        for i in vorp:
            if i.species_string not in cdict:
                cdict[i.species_string] = vorp[i]
            else:
                cdict[i.species_string] += vorp[i]
        return cdict


class ECoN(CNBase):
    """
    Effective Coordination Number (ECON) of Hoppe.
    """
    def compute(self, structure, n):
        params = self._params
        x = EffectiveCoordFinder_modified(structure, n)
        return x.get_cns(**params)


class OKeefes_mod(CNBase):
    """
    Modified O'Keefe VoronoiCoordFinder that considers only neighbors
    with at least 50% weight of max(weight).
    """
    def compute(self, structure, n):
        params = self._params
        x = OKeefes_modified(structure, n)
        return x.get_cns(**params)


class VoronoiLegacy(CNBase):
    """
    Plain Voronoi coordination numbers (i.e. number of facets of Voronoi polyhedra)
    Should not be used on its own, implemented only for comparison purposes.
    Base line for any Voronoi based CN algorithm.
    """
    def compute(self, structure, n):
        pass


class BrunnerReciprocal(CNBase):
    """
    Brunner's CN described as counting the atoms that are within the largest gap in
    differences in reciprocal interatomic distances.
    Ref:
        G.O. Brunner, A definition of coordination and its relevance in structure types AlB2 and NiAs.
        Acta Crys. A33 (1977) 226.
    """
    def compute(self, structure, n):
        params = self._params
        return Brunner(structure, n, mode="reciprocal", **params)


class BrunnerRelative(CNBase):
    """
    Brunner's CN described as counting the atoms that are within the largest gap in
    differences in real space interatomic distances.
    Note: Might be higly inaccurate in certain cases.
    Ref:
        G.O. Brunner, A definition of coordination and its relevance in structure types AlB2 and NiAs.
        Acta Crys. A33 (1977) 226.
    """
    def compute(self, structure, n):
        params = self._params
        return Brunner(structure, n, mode="relative", **params)


class BrunnerReal(CNBase):
    """
    Brunner's CN described as counting the atoms that are within the largest gap in
    differences in real space interatomic distances.
    Note: Might be higly inaccurate in certain cases.
    Ref:
        G.O. Brunner, A definition of coordination and its relevance in structure types AlB2 and NiAs.
        Acta Crys. A33 (1977) 226.
    """
    def compute(self, structure, n):
        params = self._params
        return Brunner(structure, n, mode="real", **params)


def Brunner(structure, n, mode="reciprocal", tol=1.0e-4, radius=8.0):
    """
    Helper function to compute Brunner's reciprocal gap and realspace gap CN.
    """
    nl = structure.get_neighbors(structure.sites[n], radius)
    ds = [i[-1] for i in nl]
    ds.sort()

    if mode == "reciprocal":
        ns = [1.0/ds[i] - 1.0/ds[i+1] for i in range(len(ds) - 1)]
    elif mode == "relative":
        ns = [ds[i]/ds[i+1] for i in range(len(ds) - 1)]
    elif mode == "real":
        ns = [ds[i] - ds[i+1] for i in range(len(ds) - 1)]
    else:
        raise ValueError("Unknown Brunner CN mode.")

    d_max = ds[ ns.index(max(ns)) ]
    cn = {}
    for i in nl:
        if i[-1] < d_max + tol:
            el = i[0].species_string
            if el in cn:
                cn[el] += 1.0
            else:
                cn[el] = 1.0
    return cn


class OKeefes_modified(object):
    """
    Author: S. Bajaj (LBL)
    Modified: M. Aykol (LBL)
    """
    def __init__(self, structure, n):
        self._structure = structure
        self.n = n

    def get_cns(self):
        siteno = self.n
        try:
            vor = VoronoiCoordFinder(self._structure).get_voronoi_polyhedra(siteno)
            weights = VoronoiCoordFinder(self._structure).get_voronoi_polyhedra(siteno).values()
        except RuntimeError as e:
            print e

        coordination = {}
        max_weight = max(weights)
        for v in vor:
            el = v.species_string
            if vor[v] > 0.50 * max_weight:
                if el in coordination:
                    coordination[el]+=1
                else:
                    coordination[el] = 1
        return coordination


class EffectiveCoordFinder_modified(object):

    """
    Author: S. Bajaj (LBL)
    Modified: M. Aykol (LBL)
    Finds the average effective coordination number for each cation in a given structure. It
    finds all cation-centered polyhedral in the structure, calculates the bond weight for each peripheral ion in the
    polyhedral, and sums up the bond weights to obtain the effective coordination number for each polyhedral. It then
    averages the effective coordination of all polyhedral with the same cation at the central site.
    We use the definition from Hoppe (1979) to calculate the effective coordination number of the polyhedrals:
    Hoppe, R. (1979). Effective coordination numbers (ECoN) and mean Active fictive ionic radii (MEFIR).
    Z. Kristallogr. , 150, 23-52.
    ECoN = sum(exp(1-(l_i/l_av)^6)), where l_av = sum(l_i*exp(1-(1_i/l_min)))/sum(exp(1-(1_i/l_min)))
    """

    def __init__(self, structure, n):
        self._structure = structure
        self.n = n

    def get_cns(self, radius=10.0):
        """
        Get a specie-centered polyhedra for a structure
        :param radius: (float) distance in Angstroms for bond cutoff
        :return: (dict) A dictionary with keys corresponding to different ECoN coordination numbers for site n.
        """
        site = self._structure.sites[self.n]

        all_bond_lengths = []
        neighbor_list = []
        bond_weights = []
        for neighbor in self._structure.get_neighbors(site, radius):  # entry = (site, distance)
            if neighbor[1] < radius:
                all_bond_lengths.append(neighbor[1])
                neighbor_list.append(neighbor[0].species_string)

        weighted_avg = calculate_weighted_avg(all_bond_lengths)
        cns = {}
        for i in neighbor_list:
            cns[i]=0.0

        for bond in range(len(all_bond_lengths)):
            bond_weight = math.exp(1-(all_bond_lengths[bond]/weighted_avg)**6)
            cns[ neighbor_list[bond]]+=bond_weight
        return cns


def calculate_weighted_avg(bonds):
    """
    Author: S. Bajaj (LBL)
    Get the weighted average bond length given by the effective coordination number formula in Hoppe (1979)
    :param bonds: (list) list of floats that are the bond distances between a cation and its peripheral ions
    :return: (float) exponential weighted average
    """
    minimum_bond = min(bonds)
    weighted_sum = 0.0
    total_sum = 0.0
    for entry in bonds:
        weighted_sum += entry*math.exp(1 - (entry/minimum_bond)**6)
        total_sum += math.exp(1-(entry/minimum_bond)**6)
    return weighted_sum/total_sum


if __name__ == '__main__':
    test_mpid = "mp-2534"
    with MPRester() as mp:
        test_struct = mp.get_structure_by_material_id(test_mpid)
    print get_redf(test_struct)["redf"]
    cn_methods = [OKeefes, OKeefes_mod, ECoN, BrunnerReal, BrunnerReciprocal, BrunnerRelative]
    for cn_method in cn_methods:
        print cn_method
        r = cn_method()
        print r.compute(test_struct, 0)