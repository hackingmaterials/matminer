from __future__ import division
import math
from pymatgen import MPRester
from figrecipes.plotly.make_plots import Plotly

__authors__ = 'Anubhav Jain <ajain@lbl.gov>, Saurabh Bajaj <sbajaj@lbl.gov>'


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

    Returns: tuple (rdf, dist, x) where, 'rdf' is a list containing function values, eg: 1/r_ij, 'dist' is a list
        containing distances of each neighbor from the atom being considered, and 'x' is a list of size equal to total
         number of bins and values corresponding to the sum of 'rdf' at that distance/bin.

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


if __name__ == '__main__':
    struct = MPRester().get_structure_by_material_id('mp-1')
    rdf_data = get_rdf(struct)
    print rdf_data
    Plotly().xy_plot(x_col=rdf_data.keys(), y_col=rdf_data.values())
