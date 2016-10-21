from __future__ import division, unicode_literals
import math
import numpy as np
from pymatgen import MPRester
from pymatgen.analysis.defects import ValenceIonicRadiusEvaluator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

__authors__ = 'Anubhav Jain <ajain@lbl.gov>, Saurabh Bajaj <sbajaj@lbl.gov>, ' \
              'Nils E.R. Zimmerman <nils.e.r.zimmermann@gmail.com>'


def get_packing_fraction(s):
    if not s.is_ordered:
        raise ValueError("Disordered structure support not built yet")
    total_rad = 0
    for site in s:
        total_rad += math.pi * site.specie.atomic_radius**2

    return total_rad/s.volume


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
            bin_dist = int(rij/bin_size) * bin_size
            if bin_dist in dist_rdf:
                dist_rdf[bin_dist] += 1
            else:
                dist_rdf[bin_dist] = 1
    for bin_idx in dist_rdf:
        dist_rdf[bin_idx] /= structure.density * 4 * math.pi * (bin_idx**2) * bin_size
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
        cutoff = max([np.linalg.norm(a+b+c), np.linalg.norm(-a+b+c), np.linalg.norm(a-b+c), np.linalg.norm(a+b-c)])

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


if __name__ == '__main__':
    test_mpid = "mp-2534"
    with MPRester() as mp:
        test_struct = mp.get_structure_by_material_id(test_mpid)
    print get_redf(test_struct)["redf"]
