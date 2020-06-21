from __future__ import division

import numpy as np
from pymatgen.core.composition import Composition

from matminer.featurizers.base import BaseFeaturizer
from matminer.featurizers.utils.stats import PropertyStats
from matminer.utils.data import (
    MagpieData,
    MixingEnthalpy,
)


class YangSolidSolution(BaseFeaturizer):
    """
    Mixing thermochemistry and size mismatch terms of Yang and Zhang (2012)

    This featurizer returns two different features developed by
    .. Yang and Zhang `https://linkinghub.elsevier.com/retrieve/pii/S0254058411009357`
    to predict whether metal alloys will form metallic glasses,
    crystalline solid solutions, or intermetallics.
    The first, Omega, is related to the balance between the mixing entropy and
    mixing enthalpy of the liquid phase. The second, delta, is related to the
    atomic size mismatch between the different elements of the material.

    Features
        Yang omega - Mixing thermochemistry feature, Omega
        Yang delta - Atomic size mismatch term

    References:
        .. Yang and Zhang (2012) `https://linkinghub.elsevier.com/retrieve/pii/S0254058411009357`.
    """

    def __init__(self):
        # Load in the mixing enthalpy data
        #  Creates a lookup table of the liquid mixing enthalpies
        self.dhf_mix = MixingEnthalpy()

        # Load in a table of elemental properties
        self.elem_data = MagpieData()

    def precheck(self, c: Composition) -> bool:
        """
        Precheck a single entry. YangSolidSolution does not work for compositons
        containing any binary elment combinations for which the model has no
        parameters. We can nearly equivalently approximate this by checking
        against the unary element list.

        To precheck an entire dataframe (qnd automatically gather
        the fraction of structures that will pass the precheck), please use
        precheck_dataframe.

        Args:
            c (pymatgen.Composition): The composition to precheck.

        Returns:
            (bool): If True, s passed the precheck; otherwise, it failed.
        """
        return all([e in self.dhf_mix.valid_element_list
                    for e in c.element_composition.elements])

    def featurize(self, comp):
        return [self.compute_omega(comp), self.compute_delta(comp)]

    def compute_omega(self, comp):
        """Compute Yang's mixing thermodynamics descriptor

        :math:`\\frac{T_m \Delta S_{mix}}{ |  \Delta H_{mix} | }`

        Where :math:`T_m` is average melting temperature,
        :math:`\Delta S_{mix}` is the ideal mixing entropy,
        and :math:`\Delta H_{mix}` is the average mixing enthalpies
        of all pairs of elements in the alloy

        Args:
            comp (Composition) - Composition to featurizer
        Returns:
            (float) Omega
        """

        # Special case: Elemental compound (entropy == 0 -> Omega == 1)
        if len(comp) == 1:
            return 0

        # Get the element names and fractions
        elements, fractions = zip(*comp.element_composition.fractional_composition.items())

        # Get the mean melting temperature
        mean_Tm = PropertyStats.mean(
            self.elem_data.get_elemental_properties(elements, "MeltingT"),
            fractions
        )

        # Get the mixing entropy
        entropy = np.dot(fractions, np.log(fractions)) * 8.314 / 1000

        # Get the mixing enthalpy
        enthalpy = 0
        for i, (e1, f1) in enumerate(zip(elements, fractions)):
            for e2, f2 in zip(elements[:i], fractions):
                enthalpy += f1 * f2 * self.dhf_mix.get_mixing_enthalpy(e1, e2)
        enthalpy *= 4

        # Make sure the enthalpy is nonzero
        #  The limit as dH->0 of omega is +\inf. A very small positive dH will approximate
        #  this limit without causing issues with infinite features
        enthalpy = max(1e-6, abs(enthalpy))

        return abs(mean_Tm * entropy / enthalpy)

    def compute_delta(self, comp):
        """Compute Yang's delta parameter

        :math:`\sqrt{\sum^n_{i=1} c_i \left( 1 - \\frac{r_i}{\\bar{r}} \\right)^2 }`

        where :math:`c_i` and :math:`r_i` are the fraction and radius of
        element :math:`i`, and :math:`\\bar{r}` is the fraction-weighted
        average of the radii. We use the radii compiled by
        .. Miracle et al. `https://www.tandfonline.com/doi/ref/10.1179/095066010X12646898728200?scroll=top`.

        Args:
            comp (Composition) - Composition to assess
        Returns:
            (float) delta

        """

        elements, fractions = zip(*comp.element_composition.items())

        # Get the radii of elements
        radii = self.elem_data.get_elemental_properties(elements,
                                                        "MiracleRadius")
        mean_r = PropertyStats.mean(radii, fractions)

        # Compute the mean (1 - r/\\bar{r})^2
        r_dev = np.power(1.0 - np.divide(radii, mean_r), 2)
        return np.sqrt(PropertyStats.mean(r_dev, fractions))

    def feature_labels(self):
        return ['Yang omega', 'Yang delta']

    def citations(self):
        return ["@article{Yang2012,"
                "author = {Yang, X. and Zhang, Y.},"
                "doi = {10.1016/j.matchemphys.2011.11.021},"
                "journal = {Materials Chemistry and Physics},"
                "number = {2-3},"
                "pages = {233--238},"
                "title = {{Prediction of high-entropy stabilized solid-solution in multi-component alloys}},"
                "url = {http://dx.doi.org/10.1016/j.matchemphys.2011.11.021},"
                "volume = {132},year = {2012}}"]

    def implementors(self):
        return ['Logan Ward']