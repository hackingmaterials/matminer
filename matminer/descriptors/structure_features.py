from __future__ import division
from monty import math

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