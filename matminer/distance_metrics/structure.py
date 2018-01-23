"""
THIS IS CURRENTLY JUST HOSTING LEGACY CODE.

It will soon be refactored / changed based on some discussions.

- Anubhav (12/7/17)
"""


from matminer.featurizers.site import CrystalSiteFingerprint
from matminer.featurizers.structure import SiteStatsFingerprint
import numpy as np


def get_structure_distance(s1, s2, preset_name="cn"):
    """
    Compute structure distance using an alternate (test) algorithm. Docs are
    minimal for now.
    """

    f_site = CrystalSiteFingerprint.from_preset(preset_name)
    f_structure = SiteStatsFingerprint(site_featurizer=f_site, stats=("mean",))

    f1 = f_structure.featurize(s1)
    f2 = f_structure.featurize(s2)

    return np.linalg.norm(np.array(f1) - np.array(f2))