import collections
import itertools
import os
from functools import reduce, lru_cache
from warnings import warn

import numpy as np
import pandas as pd
from pymatgen.core import Element
from pymatgen.ext.matproj import MPRester
from pymatgen.core.composition import Composition
from pymatgen.core.molecular_orbitals import MolecularOrbitals
from pymatgen.core.periodic_table import get_el_sp
from sklearn.neighbors import NearestNeighbors

from matminer.featurizers.base import BaseFeaturizer
from matminer.featurizers.utils.stats import PropertyStats
from matminer.featurizers.utils.oxidation import has_oxidation_states
from matminer.utils.data import (
    DemlData,
    MagpieData,
    PymatgenData,
    CohesiveEnergyData,
    MixingEnthalpy,
    MatscholarElementData,
    MEGNetElementData,
)

__author__ = (
    "Logan Ward, Jiming Chen, Ashwin Aggarwal, Kiran Mathew, " "Saurabh Bajaj, Qi Wang, Maxwell Dylla, Anubhav Jain"
)




def is_ionic(comp):
    """Determines whether a compound is an ionic compound.

    Looks at the oxidation states of each site and checks if both anions and cations exist

    Args:
        comp (Composition): Composition to check
    Returns:
        (bool) Whether the composition describes an ionic compound
    """

    has_cations = False
    has_anions = False

    for el in comp.elements:
        if el.oxi_state < 0:
            has_anions = True
        if el.oxi_state > 0:
            has_cations = True
        if has_anions and has_cations:
            return True
    return False


















