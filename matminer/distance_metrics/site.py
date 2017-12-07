"""
THIS IS CURRENTLY JUST HOSTING LEGACY CODE.

It will soon be refactored / changed based on some discussions.

- Anubhav (12/7/17)
"""
from matminer.featurizers.site import OPSiteFingerprint

# TODO: @nisse3000 this should be made into a Featurizer and more general than 2 classes (returns a string label for environment). Also add unit test afterward, especially since it depends on certain default for OPSiteFingerprint - AJ
def get_tet_bcc_motif(structure, idx):
    """
    Convenience class-method from Nils Zimmermann.
    Used to distinguish coordination environment in half-Heuslers.
    Args:
        structure (pymatgen Structure): the target structure to evaluate
        idx (index): the site index in the structure
    Returns:
        (str) that describes site coordination enviornment
            'bcc'
            'tet'
            'unrecognized'
    """

    op_site_fp = OPSiteFingerprint()
    fp = op_site_fp.featurize(structure, idx)
    labels = op_site_fp.feature_labels()
    i_tet = labels.index('tet CN_4')
    i_bcc = labels.index('bcc CN_8')
    if fp[i_bcc] > 0.5:
        return 'bcc'
    elif fp[i_tet] > 0.5:
        return 'tet'
    else:
        return 'unrecognized'