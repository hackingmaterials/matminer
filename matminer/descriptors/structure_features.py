from __future__ import division
import math
import pymatgen as pmg
from pymatgen import MPRester
import plotly
import plotly.graph_objs as go

__author__ = 'Anubhav Jain <ajain@lbl.gov>'


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
    rdf = []
    dist = []
    for site in structure:
        neighbors_lst = structure.get_neighbors(site, cutoff)
        for neighbor in neighbors_lst:
            rij = neighbor[1]
            rdf.append(1/(rij**2))
            dist.append(rij)
    x = [0] * int(cutoff/bin_size)   # list to
    for i, j in enumerate(dist):
        idx = int(j/bin_size)
        x[idx] += rdf[i]
    return rdf, dist, x


if __name__ == '__main__':
    struct = MPRester().get_structure_by_material_id('mp-70')
    print get_rdf(struct)
    """
        data = [go.Histogram(x=dist, xbins=dict(start=0, end=25, size=0.1))]
        fig = {'data': data}
        plotly.offline.plot(fig)
    """
