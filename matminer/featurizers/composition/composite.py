from matminer.featurizers.base import BaseFeaturizer
from matminer.featurizers.utils.stats import PropertyStats
from matminer.utils.data import (
    MagpieData,
)
from matminer.featurizers.composition.element import ElementFraction
from matminer.featurizers.composition.orbital import ValenceOrbital


class Meredig(BaseFeaturizer):
    """
    Class to calculate features as defined in Meredig et. al.

    Features:
        Atomic fraction of each of the first 103 elements, in order of atomic number.
        17 statistics of elemental properties;
            Mean atomic weight of constituent elements
            Mean periodic table row and column number
            Mean and range of atomic number
            Mean and range of atomic radius
            Mean and range of electronegativity
            Mean number of valence electrons in each orbital
            Fraction of total valence electrons in each orbital

    """

    def __init__(self):
        self.data_source = MagpieData()

        # The labels for statistics on element properties
        self._element_property_feature_labels = [
            "mean AtomicWeight",
            "mean Column",
            "mean Row",
            "range Number",
            "mean Number",
            "range AtomicRadius",
            "mean AtomicRadius",
            "range Electronegativity",
            "mean Electronegativity",
        ]
        # Initialize stats computer
        self.pstats = PropertyStats()

    def featurize(self, comp):
        """
        Get elemental property attributes

        Args:
            comp: Pymatgen composition object

        Returns:
            all_attributes: Specified property statistics of features
        """

        # First 103 features are element fractions, we can get these from the ElementFraction featurizer
        element_fraction_features = ElementFraction().featurize(comp)

        # Next 9 features are statistics on elemental properties
        elements, fractions = zip(*comp.element_composition.items())
        element_property_features = [0] * len(self._element_property_feature_labels)

        for i, feat in enumerate(self._element_property_feature_labels):
            stat = feat.split(" ")[0]
            attr = " ".join(feat.split(" ")[1:])

            elem_data = [self.data_source.get_elemental_property(e, attr) for e in elements]
            element_property_features[i] = self.pstats.calc_stat(elem_data, stat, fractions)

        # Final 8 features are statistics on valence orbitals, available from the ValenceOrbital featurizer
        valence_orbital_features = ValenceOrbital(orbitals=("s", "p", "d", "f"), props=("avg", "frac")).featurize(comp)

        return element_fraction_features + element_property_features + valence_orbital_features

    def feature_labels(self):
        # Since we have more features than just element fractions, append 'fraction' to element symbols for clarity
        element_fraction_features = [e + " fraction" for e in ElementFraction().feature_labels()]
        valence_orbital_features = ValenceOrbital().feature_labels()
        return element_fraction_features + self._element_property_feature_labels + valence_orbital_features

    def citations(self):
        citation = [
            "@article{meredig_agrawal_kirklin_saal_doak_thompson_zhang_choudhary_wolverton_2014, title={Combinatorial "
            "screening for new materials in unconstrained composition space with machine learning}, "
            "volume={89}, DOI={10.1103/PhysRevB.89.094104}, number={1}, journal={Physical "
            "Review B}, author={B. Meredig, A. Agrawal, S. Kirklin, J. E. Saal, J. W. Doak, A. Thompson, "
            "K. Zhang, A. Choudhary, and C. Wolverton}, year={2014}}"
        ]
        return citation

    def implementors(self):
        return ["Amalie Trewartha"]