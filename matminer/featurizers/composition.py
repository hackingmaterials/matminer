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






















