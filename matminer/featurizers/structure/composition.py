from __future__ import division, unicode_literals, print_function

from matminer.featurizers.base import BaseFeaturizer


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
        # NOTE Written by Logan Ward, but let's just pass
        # through the composition implementors
        return self.featurizer.implementors()
