"""
Utility functions related to pre-featurization operations, such as determining
whether a dataset is within scope of a certain featurizer or will be
computationally difficult for a particular featurizer.
"""

from copy import deepcopy

import numpy as np

__author__ = ["Alex Dunn <ardunn@lbl.gov>"]


def basic_structure_stats(structures):
    """
    Basic structure statistics related to the scope and performance of
    some structure featurizers.

    Current basic stats are the percent of ordered structures, the average
    number of sites, the maximum number of sites, and the minimum number of
    sites.

    Args:
        structures ([Structure]): An iterable of pymatgen structures

    Returns:
        stats (dict): A dictionary of statistics related to the structures.

    """
    preallocated = [None] * len(structures)
    counts = {k: deepcopy(preallocated) for k in ["n_sites", "is_ordered"]}
    for i, s in enumerate(structures):
        counts["is_ordered"][i] = s.is_ordered
        counts["n_sites"][i] = len(s.sites)
    stats = {}
    stats["fraction_ordered"] = np.sum(counts["is_ordered"]) / len(structures)
    stats["avg_n_sites"] = np.sum(counts["n_sites"]) / len(structures)
    stats["max_n_sites"] = max(counts["n_sites"])
    stats["min_n_sites"] = min(counts["n_sites"])
    return stats


def basic_composition_stats(compositions):
    """
    Basic fractional metrics based on compositions.

    Current metrics are: fraction of compositions that contain metal,
    fraction of compositions that are all metal, fraction of compositions that
    contain transition metal, fraction of compositions that are all transition
    metal, fraction of compositions that contain rare earth metals, and
    fraction of compositions that are all rare earth metals.

    Args:
        compositions ([Compositions]): An iterable of pymatgen Comnpositions.

    Returns:
        (dict): A list of fractions based on the above metrics.

    """
    preallocated = [None] * len(compositions)
    keys = [
        "fraction_contains_metal",
        "fraction_all_metal",
        "fraction_contains_transition_metal",
        "fraction_all_transition_metal",
        "fraction_contains_rare_earth_metal",
        "fraction_all_rare_earth_metal"
    ]
    counts = {k: deepcopy(preallocated) for k in keys}
    for i, c in enumerate(compositions):
        metals = [element_is_metal(e) for e in c.elements]
        counts["fraction_contains_metal"][i] = any(metals)
        counts["fraction_all_metal"][i] = all(metals)

        t_metals = [e.is_transition_metal for e in c.elements]
        counts["fraction_contains_transition_metal"][i] = any(t_metals)
        counts["fraction_all_transition_metal"][i] = all(t_metals)

        re_metals = [e.is_rare_earth_metal for e in c.elements]
        counts["fraction_contains_rare_earth_metal"][i] = any(re_metals)
        counts["fraction_all_rare_earth_metal"][i] = all(re_metals)
    return {k: np.sum(count) / len(compositions) for k, count in counts.items()}


def element_is_metal(e):
    """
    Determines whether an element is a metal.

    Args:
        e (pymatgen.Element): An element.

    Returns:
        (bool): True, if the element is a metal

    """
    return any([e.is_transition_metal, e.is_rare_earth_metal,
                e.is_alkali, e.is_alkaline, e.is_post_transition_metal])
