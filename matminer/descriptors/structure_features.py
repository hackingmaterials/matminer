from __future__ import division, unicode_literals

import os
import json
import math
import numpy as np
import itertools
from operator import itemgetter

from pymatgen.analysis.bond_valence import BV_PARAMS
from pymatgen.analysis.defects import ValenceIonicRadiusEvaluator
from pymatgen.analysis.structure_analyzer import OrderParameters
from pymatgen.core.periodic_table import Specie
from pymatgen.core.structure import Element
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

    Returns: (tuple of ndarray) first element is the normalized RDF, second is the inner radius of the RDF bin

    """
    
    if not structure.is_ordered:
        raise ValueError("Disordered structure support not built yet")
    
    # Get the distances between all atoms
    neighbors_lst = structure.get_all_neighbors(cutoff)
    all_distances = np.concatenate(tuple(map(lambda x: [itemgetter(1)(e) for e in x], neighbors_lst)))
    
    # Compute a histogram
    dist_hist, dist_bins = np.histogram(all_distances,
            bins=np.arange(0, cutoff+bin_size, bin_size), density=False)
    
    # Normalize counts
    shell_vol = 4.0 / 3.0 * math.pi * (np.power(dist_bins[1:],3) - np.power(dist_bins[:-1], 3))
    number_density = structure.num_sites / structure.volume 
    rdf = dist_hist / shell_vol / number_density
    
    return rdf, dist_bins[:-1]
    
    
def get_prdf(structure, cutoff=20.0, bin_size=0.1):
    """
    Compute the partial radial distribution function for a structure
    
    The partial radial distribution function is the radial distibution function
    broken down for each pair of atom types
    
    The PRDF was proposed as a structural descriptor by [Schutt *et al.*](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.89.205118)
    
    Args:
        structure: pymatgen structure object
        cutoff: (int/float) distance to calculate rdf up to
        bin_size: (int/float) size of bin to obtain rdf for

    Returns: (tuple) First element is a dict where keys are tuples of element names
               and values are PRDFs,
    """
    
    if not structure.is_ordered:
        raise ValueError("Disordered structure support not built yet")
        
    # Get the composition of the array
    composition = structure.composition.fractional_composition.to_reduced_dict
    
    # Get the distances between all atoms
    neighbors_lst = structure.get_all_neighbors(cutoff)
    
    # Sort neighbors by type
    distances_by_type = {}
    for p in itertools.product(composition.keys(), composition.keys()):
        distances_by_type[p] = []
        
    def get_symbol(site):
        return site.specie.symbol if isinstance(site.specie, Element) else site.specie.element.symbol
    for site,nlst in zip(structure.sites, neighbors_lst): # Each list is a list for each site
        my_elem = get_symbol(site)
        
        for neighbor in nlst:
            rij = neighbor[1]
            n_elem = get_symbol(neighbor[0])
            # LW 3May17: Any better ideas than appending each element at a time?
            distances_by_type[(my_elem,n_elem)].append(rij)
    
    # Compute and normalize the prdfs
    prdf = {}
    dist_bins = np.arange(0, cutoff+bin_size, bin_size)
    shell_volume = 4.0 / 3.0 * math.pi * (np.power(dist_bins[1:],3) - np.power(dist_bins[:-1], 3))
    for key, distances in distances_by_type.items():
        # Compute histogram of distances
        dist_hist, dist_bins = np.histogram(distances,
            bins=dist_bins, density=False)
        # Normalize
        n_alpha = composition[key[0]] * structure.num_sites    
        rdf = dist_hist / shell_volume / n_alpha
        
        prdf[key] = rdf
    
    return prdf, dist_bins[:-1]


def get_rdf_peaks(rdf, rdf_bins, n_peaks=2):
    """
    Get location of highest peaks in rdf of a structure.

    Args:
        rdf: (ndarray) as output by the function "get_rdf"
        rdf_bins: (ndarray) inner radius of the rdf bin
        n_peaks: (int) Number of the top peaks to return 

    Returns: (ndarray) of distances highest peaks, listed by descending height

    """
    
    # LW 3May17: Sorting the whole array isn't necessary, 
    #   but probably quick given typical RDF sizes are small
    return rdf_bins[np.argsort(rdf)[-n_peaks:]][::-1]


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


def get_coulomb_matrix(struct, diag_elems=False):
    """
    This function generates the Coulomb matrix, M, of the input
    structure (or molecule).  The Coulomb matrix was put forward by
    Rupp et al. (Phys. Rev. Lett. 108, 058301, 2012) and is defined by
    off-diagonal elements M_ij = Z_i*Z_j/|R_i-R_j|
    and diagonal elements 0.5*Z_i^2.4, where Z_i and R_i denote
    the nuclear charge and the position of atom i, respectively.

    Args:
        struct (Structure or Molecule): input structure (or molecule).
        diag_elems (bool): flag indicating whether (True) to use
                the original definition of the diagonal elements;
                if set to False (default),
                the diagonal elements are set to zero.
    Returns: (Nsites x Nsites matrix) Coulomb matrix.
    """
    m = [[] for s in struct.sites]
    z = []
    for s in struct.sites:
        if isinstance(s, Specie):
            z.append(Element(s.element.symbol).Z)
        else:
            z.append(Element(s.species_string).Z)
    for i in range(struct.num_sites):
        for j in range(struct.num_sites):
            if i == j:
                if diag_elems:
                    m[i].append(0.5 * z[i]**2.4)
                else:
                    m[i].append(0)
            else:
                d = struct.get_distance(i, j)
                m[i].append(z[i] * z[j] / d)
    return np.array(m)


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


def get_neighbors_of_site_with_index(struct, n, p=None):
    """
    Determine the neighbors around the site that has index n in the input
    Structure object struct, given the approach defined by parameters
    p.  All supported neighbor-finding approaches and listed and
    explained in the following.  All approaches start by creating a
    tentative list of neighbors using a large cutoff radius defined in
    parameter dictionary p via key "cutoff".
    "min_dist": find nearest neighbor and its distance d_nn; consider all
            neighbors which are within a distance of d_nn * (1 + delta),
            where delta is an additional parameter provided in the
            dictionary p via key "delta".
    "scaled_VIRE": compute the radii, r_i, of all sites on the basis of
            the valence-ionic radius evaluator (VIRE); consider all
            neighbors for which the distance to the central site is less
            than the sum of the radii multiplied by an a priori chosen
            parameter, delta,
            (i.e., dist < delta * (r_central + r_neighbor)).
    "min_relative_VIRE": same approach as "min_dist", except that we
            use relative distances (i.e., distances divided by the sum of the
            atom radii from VIRE).
    "min_relative_OKeeffe": same approach as "min_relative_VIRE", except
            that we use the bond valence parameters from O'Keeffe's bond valence
            method (J. Am. Chem. Soc. 1991, 3226-3229) to calculate
            relative distances.

    Args:
        struct (Structure): input structure.
        n (int): index of site in Structure object for which
                neighbors are to be determined.
        p (dict): specification (via "approach" key; default is "min_dist")
                and parameters of neighbor-finding approach.
                Default cutoff radius is 6 Angstrom (key: "cutoff").
                Other default parameters are as follows.
                min_dist: "delta": 0.15;
                min_relative_OKeeffe: "delta": 0.05;
                min_relative_VIRE: "delta": 0.05;
                scaled_VIRE: "delta": 2.

    Returns: ([site]) list of sites that are considered to be nearest
            neighbors to site with index n in Structure object struct.
    """
    sites = []
    if p is None:
        p = {"approach": "min_dist", "delta": 0.1,
                "cutoff": 6}

    if p["approach"] not in [
            "min_relative_OKeeffe", "min_dist", "min_relative_VIRE", \
            "scaled_VIRE"]:
        raise RuntimeError("Unsupported neighbor-finding approach"
                " (\"{}\")".format(p["approach"]))

    if p["approach"] == "min_relative_OKeeffe" or p["approach"] == "min_dist":
        neighs_dists = struct.get_neighbors(struct[n], p["cutoff"])
        try:
            eln = struct[n].specie.element
        except:
            eln = struct[n].species_string
    elif p["approach"] == "scaled_VIRE" or p["approach"] == "min_relative_VIRE":
        vire = ValenceIonicRadiusEvaluator(struct)
        if np.linalg.norm(struct[n].coords-vire.structure[n].coords) > 1e-6:
            raise RuntimeError("Mismatch between input structure and VIRE structure.")
        neighs_dists = vire.structure.get_neighbors(vire.structure[n], p["cutoff"])
        rn = vire.radii[vire.structure[n].species_string]

    reldists_neighs = []
    for neigh, dist in neighs_dists:
        if p["approach"] == "scaled_VIRE":
            dscale = p["delta"] * (vire.radii[neigh.species_string] + rn)
            if dist < dscale:
                sites.append(neigh)
        elif p["approach"] == "min_relative_VIRE":
            reldists_neighs.append([dist / (
                    vire.radii[neigh.species_string] + rn), neigh])
        elif p["approach"] == "min_relative_OKeeffe":
            try:
                el2 = neigh.specie.element
            except:
                el2 = neigh.species_string
            reldists_neighs.append([dist / get_okeeffe_distance_prediction(
                    eln, el2), neigh])
        elif p["approach"] == "min_dist":
            reldists_neighs.append([dist, neigh])

    if p["approach"] == "min_relative_VIRE" or \
            p["approach"] == "min_relative_OKeeffe" or \
            p["approach"] == "min_dist":
        min_reldist = min([reldist for reldist, neigh in reldists_neighs])
        for reldist, neigh in reldists_neighs:
            if reldist / min_reldist < 1.0 + p["delta"]:
                sites.append(neigh)

    return sites


def get_order_parameters(struct, pneighs=None, convert_none_to_zero=True):
    """
    Calculate all order parameters (OPs) for all sites in Structure object
    struct.

    Args:
        struct (Structure): input structure.
        pneighs (dict): specification and parameters of
                neighbor-finding approach (see
                get_neighbors_of_site_with_index function
                for more details).
        convert_none_to_zero (bool): flag indicating whether or not
                to convert None values in OPs to zero.

    Returns: ([[float]]) matrix of all sites' (1st dimension)
            order parameters (2nd dimension). 46 order parameters are
            computed per site: q_cn (coordination number), q_lin,
            35 x q_bent (starting with a target angle of 5 degrees and,
            increasing by 5 degrees, until 175 degrees), q_tet, q_oct,
            q_bcc, q_2, q_4, q_6, q_reg_tri, q_sq, q_sq_pyr.
    """
    opvals = []
    optypes = ["cn", "lin"]
    opparas = [[], []]
    for i in range(5, 180, 5):
        optypes.append("bent")
        opparas.append([float(i), 0.0667])
    for t in ["tet", "oct", "bcc", "q2", "q4", "q6", "reg_tri", "sq", \
            "sq_pyr"]: # , "tri_bipyr"]:
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
        struct, pneighs=None, convert_none_to_zero=True, delta_op=0.01):
    """
    Determine the order parameter statistics accumulated across all sites
    in Structure object struct using the get_order_parameters function.

    Args:
        struct (Structure): input structure.
        pneighs (dict): specification and parameters of
                neighbor-finding approach (see
                get_neighbors_of_site_with_index function
                for more details).
        convert_none_to_zero (bool): flag indicating whether or not
                to convert None values in OPs to zero (cf.,
                get_order_parameters function).
        delta_op (float): bin size of histogram that is computed
                in order to identify peak locations.

    Returns: ({}) dictionary, the keys of which represent
            the order parameter type (e.g., "bent5", "tet", "sq_pyr")
            and the values of which are dictionaries carring the
            statistics ("min", "max", "mean", "std", "peak1", "peak2").
    """
    opstats = {}
    optypes = ["cn", "lin"]
    for i in range(5, 180, 5):
        optypes.append("bent{}".format(i))
    for t in ["tet", "oct", "bcc", "q2", "q4", "q6", "reg_tri", "sq", \
            "sq_pyr"]: # , "tri_bipyr"]:
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


def site_is_of_motif_type(struct, n, pneighs=None, thresh=None):
    """
    Returns the motif type of site with index n in structure struct;
    currently featuring "tetrahedral", "octahedral", "bcc", and "cp"
    (close-packed: fcc and hcp).  If the site is not recognized
    or if it has been recognized as two different motif types,
    the function labels the site as "unrecognized".

    Args:
        struct (Structure): input structure.
        n (int): index of site in Structure object for which motif type
                is to be determined.
        pneighs (dict): specification and parameters of neighbor-finding
                approach (cf., function get_neighbors_of_site_with_index).
        thresh (dict): thresholds for motif criteria (currently, required
                keys and their default values are "qtet": 0.5,
                "qoct": 0.5, "qbcc": 0.5, "q6": 0.4).

    Returns: motif type (str).
    """

    if thresh is None:
        thresh = {
                "qtet": 0.5, "qoct": 0.5, "qbcc": 0.5, "q6": 0.4,
                "qtribipyr": 0.8}

    ops = get_order_parameters(struct, pneighs=pneighs)
    cn = int(ops[n][0] + 0.5)
    motif_type = "unrecognized"
    nmotif = 0

    if cn == 4 and ops[n][37] > thresh["qtet"]:
        motif_type = "tetrahedral"
        nmotif += 1
    #if cn == 5 and ops[n][46] > thresh["qtribipyr"]:
    #    motif_type = "trigonal bipyramidal"
    #    nmotif += 1
    if cn == 6 and ops[n][38] > thresh["qoct"]:
        motif_type = "octahedral"
        nmotif += 1
    if cn == 8 and (ops[n][39] > thresh["qbcc"] and \
            ops[n][37] < thresh["qtet"]):
        motif_type = "bcc"
        nmotif += 1
    if cn == 12 and (ops[n][42] > thresh["q6"] and \
            ops[n][37] < thresh["q6"] and \
            ops[n][38] < thresh["q6"] and \
            ops[n][39] < thresh["q6"]):
        motif_type = "cp"
        nmotif += 1

    if nmotif > 1:
        motif_type = "unrecognized"

    return motif_type

def get_okeeffe_params(el_symbol):
    """
    Returns the elemental parameters related to atom size and
    electronegativity which are used for estimating bond-valence
    parameters (bond length) of pairs of atoms on the basis of data
    provided in 'Atoms Sizes and Bond Lengths in Molecules and Crystals'
    (O'Keeffe & Brese, 1991).

    Args:
        el_symbol (str): element symbol.
    Returns:
        (dict): atom-size ('r') and electronegativity-related ('c')
                parameter.
    """

    el = Element(el_symbol)
    if el not in list(BV_PARAMS.keys()):
        raise RuntimeError("Could not find O'Keeffe parameters for element"
                " \"{}\" in \"BV_PARAMS\"dictonary"
                " provided by pymatgen".format(el_symbol))

    return BV_PARAMS[el]


def get_okeeffe_distance_prediction(el1, el2):
    """
    Returns an estimate of the bond valence parameter (bond length) using
    the derived parameters from 'Atoms Sizes and Bond Lengths in Molecules
    and Crystals' (O'Keeffe & Brese, 1991). The estimate is based on two
    experimental parameters: r and c. The value for r  is based off radius,
    while c is (usually) the Allred-Rochow electronegativity. Values used
    are *not* generated from pymatgen, and are found in
    'okeeffe_params.json'.

    Args:
        el1, el2 (Element): two Element objects
    Returns:
        a float value of the predicted bond length
    """
    el1_okeeffe_params = get_okeeffe_params(el1)
    el2_okeeffe_params = get_okeeffe_params(el2)

    r1 = el1_okeeffe_params['r']
    r2 = el2_okeeffe_params['r']
    c1 = el1_okeeffe_params['c']
    c2 = el2_okeeffe_params['c']

    return r1 + r2 - r1*r2*math.pow(math.sqrt(c1)-math.sqrt(c2), 2)/(c1*r1+c2*r2)
