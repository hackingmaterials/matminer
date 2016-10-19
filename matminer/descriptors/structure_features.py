from __future__ import division, unicode_literals
import math
import numpy as np
from pymatgen import MPRester
from figrecipes.plotly.make_plots import Plotly
from pymatgen.analysis.defects import ValenceIonicRadiusEvaluator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

__authors__ = 'Anubhav Jain <ajain@lbl.gov>, Saurabh Bajaj <sbajaj@lbl.gov>, ' \
              'Nils E.R. Zimmerman <nils.e.r.zimmermann@gmail.com>'


def get_packing_fraction(s):
    if not s.is_ordered:
        raise ValueError("Disordered structure support not built yet")
    total_rad = 0
    for site in s:
        total_rad += site.specie.atomic_radius**2 * math.pi

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

# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.


class ElectronicRadialDistributionFunction(object):
    """
    This class permits the
    """

    def get_redf(self, struct, cutoff=-1.0, dr=0.05):
        """
        calculation of the crystal structure-inherent electronic radial
        distribution function (ReDF) according to Willighagen et al.,
        Acta Cryst., 2005, B61, 29-36. The ReDF is a structure-integral RDF
        (i.e., summed over all sites) in which the positions of neighboring
        sites are weighted by electrostatic interactions inferred from atomic 
        partial charges. Atomic charges are obtained from the
        ValenceIonicRadiusEvaluator class.

        Args:
            struct (Structure): input Structure object.
            cutoff (float): distance up to which the ReDF is to be
                    calculated (default: -1.0; will trigger
                    using length of longest diagonal in primitive
                    unit cell).
            dr (float): width of bins ("x"-axis) of ReDF (default: 0.05 A).
        """
        prim_struct = SpacegroupAnalyzer(struct).find_primitive()
        if prim_struct == None:
            prim_struct = struct.copy()
        valrad_eval = ValenceIonicRadiusEvaluator(prim_struct)
        prim_struct = valrad_eval.structure
        val = valrad_eval.valences

        if cutoff <= 0.0:
            a = prim_struct.lattice.matrix[0]
            b = prim_struct.lattice.matrix[1]
            c = prim_struct.lattice.matrix[2]
            cutoff = max([np.linalg.norm(a+b+c), np.linalg.norm(-a+b+c), \
                    np.linalg.norm(a-b+c), np.linalg.norm(a+b-c)])

        if dr <= 0.0:
            raise RuntimeError("width of bins for ReDF"
                    " must be larger than zero.")

        nbins = int(cutoff / dr) + 1
        natoms_f = float(prim_struct.num_sites)

        data = {}
        data["distances"] = np.array(
                [(float(i) + 0.5) * dr for i in range(nbins)])
        data["redf"] = np.zeros(nbins, dtype=np.float)

        for site in prim_struct.sites:
            this_charge = float(site.specie.oxi_state)
            neighs_dists = prim_struct.get_neighbors(site, cutoff)
            for neigh, dist in neighs_dists:
                neigh_charge = float(neigh.specie.oxi_state)
                s = int(dist / dr)
                data["redf"][s] = data["redf"][s] + \
                        (this_charge * neigh_charge) / (natoms_f * dist)


if __name__ == '__main__':
    struct = MPRester().get_structure_by_material_id('mp-1')
    rdf_data = get_rdf(struct)
    print rdf_data
    Plotly().xy_plot(x_col=rdf_data.keys(), y_col=rdf_data.values())
    print get_rdf_peaks(rdf_data)
