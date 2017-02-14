from __future__ import division, unicode_literals

import math
import numpy as np

from pymatgen.analysis.defects import ValenceIonicRadiusEvaluator
from pymatgen.analysis.structure_analyzer import OrderParameters
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


__authors__ = 'Anubhav Jain <ajain@lbl.gov>, Saurabh Bajaj <sbajaj@lbl.gov>, ' \
              'Nils E.R. Zimmerman <nils.e.r.zimmermann@gmail.com>'


def get_packing_fraction(s):
    if not s.is_ordered:
        raise ValueError("Disordered structure support not built yet")
    total_rad = 0
    for site in s:
        total_rad += site.specie.atomic_radius**3

    return 4 * math.pi * total_rad / (3 * s.volume)


def get_vol_per_site(s):
    if not s.is_ordered:
        raise ValueError("Disordered structure support not built yet")

    return s.volume / len(s)


def get_density(s):
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
    distances = list(dist_rdf.keys())
    rdf = list(dist_rdf.values())
    sorted_rdfs = sorted(rdf, reverse=True)
    max_rdf, second_highest_rdf = sorted_rdfs[0], sorted_rdfs[1]
    max_idx = rdf.index(max_rdf)
    second_highest_idx = rdf.index(second_highest_rdf)
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


def get_min_relative_distances(struct, cutoff=10.0):
    """
    This function determines the relative distance of each site to its closest
    neighbor. We use the relative distance,
    f_ij = r_ij / (r^atom_i + r^atom_j), as a measure rather than the absolute
    distances, r_ij, to account for the fact that different atoms/species
    have different sizes.  The function uses the valence-ionic radius
    estimator implemented in pymatgen.

    Args:
        struct (Structure): input structure.
        cutoff (float): (absolute) distance up to which tentative closest
                neighbors (on the basis of relative distances) are
                to be determined.

    Returns: ([float]) list of all minimum relative distances (i.e., for all
        sites).
    """
    vire = ValenceIonicRadiusEvaluator(struct)
    min_rel_dists = []
    for site in vire.structure:
        min_rel_dists.append(min([dist/(vire.radii[site.species_string]+\
            vire.radii[neigh.species_string]) for neigh, dist in \
            vire.structure.get_neighbors(site, cutoff)]))
    return min_rel_dists[:]


def get_neighbors_of_site_with_index(struct, n, p={}):
    """
    Determine the neighbors around the site that has index n in the input
    Structure object struct, given a pre-defined approach.  So far,
    "scaled_VIRE" and "min_relative_VIRE" are implemented
    (VIRE = valence-ionic radius evaluator).

    Args:
        struct (Structure): input structure.
        n (int): index of site in Structure object for which
                neighbors are to be determined.
        p (dict): specification ("approach") and parameters of
                neighbor-finding approach.
                min_relative_VIRE (default): "delta_scale" (0.05)
                and "scale_cut" (4);
                scaled_VIRE: "scale" (2) and "scale_cut" (4).

    Returns: ([site]) list of sites that are considered to be nearest
            neighbors to site with index n in Structure object struct.
    """
    sites = []
    if p == {}:
        p = {"approach": "min_relative_VIRE", "delta_scale": 0.05,
                "scale_cut": 4}
    if p["approach"] == "scaled_VIRE" or p["approach"] == "min_relative_VIRE":
        vire = ValenceIonicRadiusEvaluator(struct)
        if np.linalg.norm(struct[n].coords-vire.structure[n].coords) > 1e-6:
            raise RuntimeError("Mismatch between input structure and VIRE structure.")
        maxr = p["scale_cut"] * 2.0 * max(vire.radii.values())
        neighs_dists = vire.structure.get_neighbors(vire.structure[n], maxr)
        #print(len(neighs_dists))
        rn = vire.radii[vire.structure[n].species_string]
        #print(str(vire.structure[n]))
        reldists_neighs = []
        for neigh, dist in neighs_dists:
            if p["approach"] == "scaled_VIRE":
                dscale = p["scale"] * (vire.radii[neigh.species_string] + rn)
                #print("{} {}".format(dist, dscale))
                if dist < dscale:
                    sites.append(neigh)
                    #print(str(neigh))
            else:
                reldists_neighs.append([dist/(
                        vire.radii[neigh.species_string] + rn), neigh])
        if p["approach"] == "min_relative_VIRE":
            reldists_neighs_sorted = sorted(reldists_neighs)
            max_reldist = reldists_neighs_sorted[0][0] + p["delta_scale"]
            for reldist, neigh in reldists_neighs_sorted:
                if reldist < max_reldist:
                    sites.append(neigh)
                else:
                    break
            #for reldist, neigh in reldists_neighs_sorted:
            #    print(str(reldist))
            #print(str(sites))
    else:
        raise RuntimeError("Unsupported neighbor-finding approach"
                " (\"{}\")".format(p["approach"]))
    return sites


def get_order_parameters(struct, pneighs={}, convert_none_to_zero=True):
    """
    Determine the neighbors around the site that has index n in the input
    Structure object struct, given a pre-defined approach.  So far,
    "scaled_VIRE" and "min_relative_VIRE" are implemented
    (VIRE = valence-ionic radius evaluator).

    Args:
        struct (Structure): input structure.
        pneighs (dict): specification ("approach") and parameters of
                neighbor-finding approach (see
                get_neighbors_of_site_with_index function
                for more details; default: min_relative_VIRE,
                delta_scale = 0.05, scale_cut = 4).
        convert_none_to_zero (bool): flag indicating whether or not
                to convert None values in OPs to zero.

    Returns: ([[float]]) matrix of all sites' (1st dimension)
            order parameters (2nd dimension). 46 order parameters are
            computed per site: q_cn (coordination number), q_lin,
            35 x q_bent (starting with a target angle of 5 degrees and,
            increasing by 5 degrees, until 175 degrees), q_tet, q_oct,
            q_bcc, q_2, q_4, q_6, q_reg_tri, q_sq, q_sq_pyr.
    """
    if pneighs == {}:
        pneighs = {"approach": "min_relative_VIRE", "delta_scale": 0.05,
                "scale_cut": 4}
    opvals = []
    optypes = ["cn", "lin"]
    opparas = [[], []]
    for i in range(5, 180, 5):
        optypes.append("bent")
        opparas.append([float(i), 0.0667])
    for t in ["tet", "oct", "bcc", "q2", "q4", "q6", "reg_tri", "sq", \
            "sq_pyr"]:
        optypes.append(t)
        opparas.append([])
    ops = OrderParameters(optypes, opparas, 100.0)
    for i, s in enumerate(struct.sites):
        neighcent = get_neighbors_of_site_with_index(struct, i, pneighs)
        neighcent.append(s)
        opvals.append(ops.get_order_parameters(
                neighcent, len(neighcent)-1,
                indeces_neighs=[j for j in range(len(neighcent)-1)]))
        if convert_none_to_zero:
            for j, opval in enumerate(opvals[i]):
                if opval is None:
                    opvals[i][j] = 0.0
    return opvals

def get_order_parameter_stats(
        struct, pneighs={}, convert_none_to_zero=True, delta_op=0.01):
    """
    Determine the order parameter statistics based on the
    data from the get_order_parameters function.

    Args:
        struct (Structure): input structure.
        pneighs (dict): specification ("approach") and parameters of
                neighbor-finding approach (see
                get_neighbors_of_site_with_index function
                for more details; default: min_relative_VIRE,
                delta_scale = 0.05, scale_cut = 4).
        convert_none_to_zero (bool): flag indicating whether or not
                to convert None values in OPs to zero.
        delta_op (float): bin size of histogram that is computed
                in order to identify peak locations.

    Returns: ({}) dictionary, the keys of which represent
            the order parameter type (e.g., "bent5", "tet", "sq_pyr")
            and the values are another dictionary carring the
            statistics ("min", "max", "mean", "std", "peak1", "peak2").
    """
    opstats = {}
    optypes = ["cn", "lin"]
    for i in range(5, 180, 5):
        optypes.append("bent{}".format(i))
    for t in ["tet", "oct", "bcc", "q2", "q4", "q6", "reg_tri", "sq", \
            "sq_pyr"]:
        optypes.append(t)
    opvals = get_order_parameters(
            struct, pneighs=pneighs, convert_none_to_zero=convert_none_to_zero)
    opvals2 = [[] for t in optypes]
    for i, opsite in enumerate(opvals):
        for j, op in enumerate(opsite):
            opvals2[j].append(op)
    for i, opstype in enumerate(opvals2):
        ops_hist = {}
        for op in opstype:
            b = round(op / delta_op) * delta_op
            if b in ops_hist.keys():
                ops_hist[b] += 1
            else:
                ops_hist[b] = 1
        ops =list(ops_hist.keys())
        hist = list(ops_hist.values())
        sorted_hist = sorted(hist, reverse=True)
        if len(sorted_hist) > 1:
            max1_hist, max2_hist = sorted_hist[0], sorted_hist[1]
        elif len(sorted_hist) > 0:
            max1_hist, max2_hist = sorted_hist[0], sorted_hist[0]
        else:
            raise RuntimeError("Could not compute OP histogram.")
        max1_idx = hist.index(max1_hist)
        max2_idx = hist.index(max2_hist)
        opstats[optypes[i]] = {
                "min": min(opstype),
                "max": max(opstype),
                "mean": np.mean(np.array(opstype)),
                "std": np.std(np.array(opstype)),
                "peak1": ops[max1_idx],
                "peak2": ops[max2_idx]}
    return opstats
