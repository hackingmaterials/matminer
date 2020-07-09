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
# Core Featurizers
# --> Radial site features
from matminer.featurizers.site.radial import (
    GeneralizedRadialDistributionFunction,
    AGNIFingerprints,
)
# --> Angular site features
from matminer.featurizers.site.angular import (
    BondOrientationalParameter,
)
# --> Angular and radial density features
from matminer.featurizers.site.density import (
    # NOTE comprhys: maybe rename BehlerParrinelloSymmetryFunctions
    # as most atoms density methods use gaussians somehow.
    GaussianSymmFunc,
    AngularFourierSeries,
)
# --> Chemical environment based features
from matminer.featurizers.site.chemical import (
    ChemEnvSiteFingerprint,
    ChemicalSRO,
)
# --> Element property based features
from matminer.featurizers.site.properties import (
    SiteElementalProperty,
    LocalPropertyDifference,
)
# --> Nearest Neighbour based features
from matminer.featurizers.site.neighbours import (
    AverageBondAngle,
    AverageBondLength,
    CoordinationNumber,
    OPSiteFingerprint,
    VoronoiFingerprint,
    CrystalNNFingerprint,
    cn_target_motif_op,  # needed for test
    cn_motif_op_params,  # needed for test
)
# --> Miscellaneous features
from matminer.featurizers.site.misc import (
    EwaldSiteEnergy,
    IntersticeDistribution,
)

# External Featurizers
# --> Featurizers using dscribe (https://singroup.github.io/dscribe/)
from matminer.featurizers.site.external.dscribe import (
    SOAP
)

# NOTE comprhys: site.py was missing "__author__"