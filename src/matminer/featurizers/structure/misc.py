"""
Miscellaneous structure featurizers.
"""

import numpy as np
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.analysis.ewald import EwaldSummation
from scipy.stats import gaussian_kde

from matminer.featurizers.base import BaseFeaturizer


class EwaldEnergy(BaseFeaturizer):
    """
    Compute the energy from Coulombic interactions.

    Note: The energy is computed using _charges already defined for the structure_.

    Features:
        ewald_energy - Coulomb interaction energy of the structure"""

    def __init__(self, accuracy=4, per_atom=True):
        """
        Args:
            accuracy (int): Accuracy of Ewald summation, number of decimal places
        """
        self.accuracy = accuracy
        self.per_atom = per_atom

    def featurize(self, strc):
        """

        Args:
             (Structure) - Structure being analyzed
        Returns:
            ([float]) - Electrostatic energy of the structure
        """
        # Compute the total energy
        ewald = EwaldSummation(strc, acc_factor=self.accuracy)
        return [ewald.total_energy / len(strc)] if self.per_atom else [ewald.total_energy]

    def feature_labels(self):
        return ["ewald_energy_per_atom"] if self.per_atom else ["ewald_energy"]

    def implementors(self):
        return ["Logan Ward", "Anubhav Jain"]

    def citations(self):
        return [
            "@Article{Ewald1921,"
            "author = {Ewald, P. P.},"
            "doi = {10.1002/andp.19213690304},"
            "issn = {00033804},"
            "journal = {Annalen der Physik},"
            "number = {3},"
            "pages = {253--287},"
            "title = {{Die Berechnung optischer und elektrostatischer "
            "Gitterpotentiale}},"
            "url = {http://doi.wiley.com/10.1002/andp.19213690304},"
            "volume = {369},"
            "year = {1921}"
            "}"
        ]


class StructureComposition(BaseFeaturizer):
    """
    Features related to the composition of a structure

    This class is just a wrapper that calls a composition-based featurizer
    on the composition of a Structure

    Features:
        - Depends on the featurizer
    """

    def __init__(self, featurizer=None):
        """Initialize the featurizer

        Args:
            featurizer (BaseFeaturizer) - Composition-based featurizer
        """
        self.featurizer = featurizer

    def fit(self, X, y=None, **fit_kwargs):
        # Get the compositions of each of the structures
        comps = [x.composition for x in X]

        return self.featurizer.fit(comps, y, **fit_kwargs)

    def featurize(self, strc):
        return self.featurizer.featurize(strc.composition)

    def feature_labels(self):
        return self.featurizer.feature_labels()

    def citations(self):
        return self.featurizer.citations()

    def implementors(self):
        # Written by Logan Ward, but let's just pass through the
        #  composition implementors
        return self.featurizer.implementors()


class XRDPowderPattern(BaseFeaturizer):
    """
    1D array representing powder diffraction of a structure as calculated by
    pymatgen. The powder is smeared / normalized according to gaussian_kde.
    """

    def __init__(self, two_theta_range=(0, 127), bw_method=0.05, pattern_length=None, **kwargs):
        """
        Initialize the featurizer.

        Args:
            two_theta_range ([float of length 2]): Tuple for range of
                two_thetas to calculate in degrees. Defaults to (0, 90). Set to
                None if you want all diffracted beams within the limiting
                sphere of radius 2 / wavelength.
            bw_method (float): how much to smear the XRD pattern
            pattern_length (float): length of final array; defaults to one value
             per degree (i.e. two_theta_range + 1)
            **kwargs: any other arguments to pass into pymatgen's XRDCalculator,
                such as the type of radiation.
        """
        self.two_theta_range = two_theta_range
        self.bw_method = bw_method
        self.pattern_length = pattern_length or two_theta_range[1] - two_theta_range[0] + 1
        self.xrd_calc = XRDCalculator(**kwargs)

    def featurize(self, strc):
        pattern = self.xrd_calc.get_pattern(strc, two_theta_range=self.two_theta_range)
        x, y = pattern.x, pattern.y
        hist = []
        for x1, y1 in zip(x, y):
            num = int(y1)
            hist += [x1] * num

        kernel = gaussian_kde(hist, bw_method=self.bw_method)
        x = np.linspace(self.two_theta_range[0], self.two_theta_range[1], self.pattern_length)
        y = kernel(x)

        return y

    def feature_labels(self):
        return [f"xrd_{x}" for x in range(self.pattern_length)]

    def citations(self):
        return [
            "@article{Ong2013, author = {Ong, Shyue Ping and Richards, "
            "William Davidson and Jain, Anubhav and Hautier, "
            "Geoffroy and Kocher, Michael and Cholia, Shreyas and Gunter, "
            "Dan and Chevrier, Vincent L. and Persson, "
            "Kristin A. and Ceder, Gerbrand}, "
            "doi = {10.1016/j.commatsci.2012.10.028}, issn = {09270256}, "
            "journal = {Computational Materials Science}, month = {feb}, "
            "pages = {314--319}, "
            "publisher = {Elsevier B.V.}, title = {{Python Materials "
            "Genomics (pymatgen): A robust, open-source python "
            "library for materials analysis}}, url = "
            "{http://linkinghub.elsevier.com/retrieve/pii/S0927025612006295}, "
            "volume = {68}, year = {2013} } "
        ]

    def implementors(self):
        return ["Anubhav Jain", "Matthew Horton"]
