import copy
from functools import lru_cache
from monty.dev import requires
from pymatgen.io.ase import AseAtomsAdaptor
from sklearn.exceptions import NotFittedError

from matminer.featurizers.utils.grdf import Gaussian, Histogram
from matminer.utils.caching import get_nearest_neighbors
from matminer.utils.data import MagpieData

"""
Features that describe the local environment of a single atom. Note that
structural features can be constructed from a combination of site features from
every site in the structure.

The `featurize` function takes two arguments:
    struct (Structure): Object representing the structure containing the site
        of interest
    idx (int): Index of the site to be featurized
We have to use two parameters because the Site object does not hold a pointer
back to its structure and often information on neighbors is required. To run
:code:`featurize_dataframe`, you must pass the column names for both the site
index and the structure. For example:
.. code:: python
    f = AGNIFingerprints()
    f.featurize_dataframe(data, ['structure', 'site_idx'])
"""

import os
import warnings
import ruamel.yaml as yaml
import itertools
import numpy as np
import scipy.integrate as integrate

from matminer.featurizers.base import BaseFeaturizer
from math import pi
from scipy.special import sph_harm
from scipy.spatial import ConvexHull
from sympy.physics.wigner import wigner_3j
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.local_env import (
    LocalStructOrderParams,
    VoronoiNN,
    CrystalNN,
    solid_angle,
    vol_tetra,
)
import pymatgen.analysis.local_env
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import (
    LocalGeometryFinder,
)
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import (
    SimplestChemenvStrategy,
    MultiWeightsChemenvStrategy,
)

from matminer.featurizers.utils.stats import PropertyStats
from sklearn.utils.validation import check_is_fitted

# SOAPFeaturizer
try:
    import dscribe
    from dscribe.descriptors import SOAP as SOAP_dscribe
except ImportError:
    dscribe, SOAP_dscribe = None, None

cn_motif_op_params = {}
with open(
    os.path.join(os.path.dirname(pymatgen.analysis.local_env.__file__), "cn_opt_params.yaml"),
    "r",
) as f:
    cn_motif_op_params = yaml.safe_load(f)
cn_target_motif_op = {}
with open(os.path.join(os.path.dirname(__file__), "cn_target_motif_op.yaml"), "r") as f:
    cn_target_motif_op = yaml.safe_load(f)