"""
Structure featurizers based on symmetry.
"""
import pymatgen.analysis.local_env as pmg_le
from pymatgen.analysis.dimensionality import get_dimensionality_larsen
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from matminer.featurizers.base import BaseFeaturizer


class GlobalSymmetryFeatures(BaseFeaturizer):
    """
    Determines symmetry features, e.g. spacegroup number and  crystal system

    Features:
        - Spacegroup number
        - Crystal system (1 of 7)
        - Centrosymmetry (has inversion symmetry)
        - Number of symmetry ops, obtained from the spacegroup
    """

    crystal_idx = {
        "triclinic": 7,
        "monoclinic": 6,
        "orthorhombic": 5,
        "tetragonal": 4,
        "trigonal": 3,
        "hexagonal": 2,
        "cubic": 1,
    }

    all_features = ["spacegroup_num", "crystal_system", "crystal_system_int", "is_centrosymmetric", "n_symmetry_ops"]

    def __init__(self, desired_features=None):
        self.features = desired_features if desired_features else self.all_features

    def featurize(self, s):
        sga = SpacegroupAnalyzer(s)
        output = []

        if "spacegroup_num" in self.features:
            output.append(sga.get_space_group_number())

        if "crystal_system" in self.features:
            output.append(sga.get_crystal_system())

        if "crystal_system_int" in self.features:
            output.append(GlobalSymmetryFeatures.crystal_idx[sga.get_crystal_system()])

        if "is_centrosymmetric" in self.features:
            output.append(sga.is_laue())

        if "n_symmetry_ops" in self.features:
            output.append(len(sga.get_symmetry_operations()))

        return output

    def feature_labels(self):
        return [x for x in self.all_features if x in self.features]

    def citations(self):
        return []

    def implementors(self):
        return ["Anubhav Jain", "Alex Dunn"]


class Dimensionality(BaseFeaturizer):
    """
    Returns dimensionality of structure: 1 means linear chains of atoms OR
    isolated atoms/no bonds, 2 means layered, 3 means 3D connected
    structure. This feature is sensitive to bond length tables that you use.
    """

    def __init__(self, nn_method=pmg_le.CrystalNN()):
        """

        Args:
            **nn_method: The nearest neighbor method used to determine atomic
                connectivity.
        """
        self.nn_method = nn_method

    def featurize(self, s):
        bs = self.nn_method.get_bonded_structure(s)
        return [get_dimensionality_larsen(bs)]

    def feature_labels(self):
        return ["dimensionality"]

    def citations(self):
        return [
            "@article{larsen2019definition, title={Definition of a scoring "
            "parameter to identify low-dimensional materials components},"
            "author={Larsen, Peter Mahler and Pandey, Mohnish and Strange, "
            "Mikkel and Jacobsen, Karsten Wedel}, journal={Physical Review "
            "Materials}, volume={3}, number={3}, pages={034003}, "
            "year={2019}, publisher={APS} }"
        ]

    def implementors(self):
        return ["Anubhav Jain", "Alex Ganose"]
