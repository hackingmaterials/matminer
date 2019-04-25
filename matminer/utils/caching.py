"""Provides utility functions for caching the results of expensive operations,
such as determining the nearest neighbors of atoms in a structure"""

from functools import lru_cache

from pymatgen.core.structure import IStructure, Structure


def get_nearest_neighbors(method, structure, site_idx):
    """Get the nearest neighbor list of a particular site in a structure

    Args:
        method (NearNeighbor) - Method used to compute nearest neighbors
        structure (Structure) - Structure to study
        site_idx (int) - Index of site to study
    Returns:
        Output of `method.get_nn_info(structure, site_idx)`
    """
    return get_all_nearest_neighbors(method, structure)[site_idx]


def get_all_nearest_neighbors(method, structure):
    """Get the nearest neighbor list of a structure

    Args:
        method (NearNeighbor) - Method used to compute nearest neighbors
        structure (IStructure) - Structure to study
    Returns:
        Output of `method.get_all_nn_info(structure)`
    """

    # pymatgen does not hash Structure objects, so we need
    #  to convert from Structure to the immutatble IStructure method
    if isinstance(structure, Structure):
        structure = IStructure.from_sites(structure)
    return _get_all_nearest_neighbors(method, structure)


@lru_cache(maxsize=1)
def _get_all_nearest_neighbors(method, structure):
    """Get the nearest neighbor list of a structure

    Args:
        method (NearNeighbor) - Method used to compute nearest neighbors
        structure (IStructure) - Structure to study
    Returns:
        Output of `method.get_all_nn_info(structure)`
    """
    return method.get_all_nn_info(structure)