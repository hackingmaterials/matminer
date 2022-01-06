"""
Composition featurizers for compositions with ionic data.
"""
import itertools

import numpy as np

from matminer.featurizers.base import BaseFeaturizer
from matminer.featurizers.composition.composite import ElementProperty
from matminer.featurizers.utils.oxidation import has_oxidation_states
from matminer.featurizers.utils.stats import PropertyStats
from matminer.utils.data import DemlData, PymatgenData


class CationProperty(ElementProperty):
    """
    Features based on properties of cations in a material

    Requires that oxidation states have already been determined. Property
    statistics weighted by composition.

    Features: Based on the statistics of the data_source chosen, computed
    by element stoichiometry. The format generally is:

    "{data source} {statistic} {property}"

    For example:

    "DemlData range magn_moment" # Range of magnetic moment via Deml et al. data

    For a list of all statistics, see the PropertyStats documentation; for a
    list of all attributes available for a given data_source, see the
    documentation for the data sources (e.g., PymatgenData, MagpieData,
    MatscholarElementData, etc.).
    """

    @classmethod
    def from_preset(cls, preset_name):
        if preset_name == "deml":
            data_source = "deml"
            features = [
                "total_ioniz",
                "xtal_field_split",
                "magn_moment",
                "so_coupling",
                "sat_magn",
            ]
            stats = ["minimum", "maximum", "range", "mean", "std_dev"]
        else:
            raise ValueError('Preset "%s" not found' % preset_name)
        return cls(data_source, features, stats)

    def feature_labels(self):
        return [f + " of cations" for f in super().feature_labels()]

    def featurize(self, comp):
        # Check if oxidation states are present
        if not has_oxidation_states(comp):
            raise ValueError("Oxidation states have not been determined")
        if not is_ionic(comp):
            raise ValueError("Composition is not ionic")

        # Prepare to store the attributes
        all_attributes = []

        # Initialize stats computer
        pstats = PropertyStats()

        # Get the cation species and fractions
        cations, fractions = zip(*((s, f) for s, f in comp.items() if s.oxi_state > 0))

        for attr in self.features:
            elem_data = [self.data_source.get_charge_dependent_property_from_specie(c, attr) for c in cations]

            for stat in self.stats:
                all_attributes.append(pstats.calc_stat(elem_data, stat, fractions))

        return all_attributes

    def citations(self):
        return [
            "@article{deml_ohayre_wolverton_stevanovic_2016, title={Predicting density "
            "functional theory total energies and enthalpies of formation of metal-nonmetal "
            "compounds by linear regression}, volume={47}, DOI={10.1002/chin.201644254}, "
            "number={44}, journal={ChemInform}, author={Deml, Ann M. and Ohayre, Ryan and "
            "Wolverton, Chris and Stevanovic, Vladan}, year={2016}}"
        ]


class OxidationStates(BaseFeaturizer):
    """
    Statistics about the oxidation states for each specie.
    Features are concentration-weighted statistics of the oxidation states.
    """

    def __init__(self, stats=None):
        """

        Args:
             stats - (list of string), which statistics compute
        """
        self.stats = stats or ["minimum", "maximum", "range", "std_dev"]

    @classmethod
    def from_preset(cls, preset_name):
        if preset_name == "deml":
            stats = ["minimum", "maximum", "range", "std_dev"]
        else:
            ValueError('Preset "%s" not found' % preset_name)
        return cls(stats=stats)

    def featurize(self, comp):
        # Check if oxidation states are present
        if not has_oxidation_states(comp):
            raise ValueError("Oxidation states have not been determined")

        # Get the oxidation states and their proportions
        oxid_states, fractions = zip(*((s.oxi_state, f) for s, f in comp.items()))

        # Compute statistics
        return [PropertyStats.calc_stat(oxid_states, s, fractions) for s in self.stats]

    def feature_labels(self):
        return ["%s oxidation state" % s for s in self.stats]

    def citations(self):
        return [
            "@article{deml_ohayre_wolverton_stevanovic_2016, title={Predicting density "
            "functional theory total energies and enthalpies of formation of metal-nonmetal "
            "compounds by linear regression}, volume={47}, DOI={10.1002/chin.201644254}, "
            "number={44}, journal={ChemInform}, author={Deml, Ann M. and Ohayre, Ryan and "
            "Wolverton, Chris and Stevanovic, Vladan}, year={2016}}"
        ]

    def implementors(self):
        return ["Logan Ward"]


class IonProperty(BaseFeaturizer):
    """
    Ionic property attributes. Similar to ElementProperty.
    """

    def __init__(self, data_source=PymatgenData(), fast=False):
        """

        Args:
             data_source - (OxidationStateMixin) - A AbstractData class that supports
                the `get_oxidation_state` method.
            fast - (boolean) whether to assume elements exist in a single oxidation state,
                which can dramatically accelerate the calculation of whether an ionic compound
                is possible, but will miss heterovalent compounds like Fe3O4.
        """
        self.data_source = data_source
        self.fast = fast

    def featurize(self, comp):
        """
        Ionic character attributes

        Args:
            comp: (Composition) Composition to be featurized

        Returns:
            cpd_possible (bool): Indicates if a neutral ionic compound is possible
            max_ionic_char (float): Maximum ionic character between two atoms
            avg_ionic_char (float): Average ionic character
        """

        elements, fractions = zip(*comp.element_composition.items())

        if len(elements) < 2:  # Single element
            cpd_possible = True
            max_ionic_char = 0
            avg_ionic_char = 0
        else:
            # Get magpie data for each element
            elec = self.data_source.get_elemental_properties(elements, "X")

            # Determine if neutral compound is possible
            if has_oxidation_states(comp):
                charges, fractions = zip(*((s.oxi_state, f) for s, f in comp.items()))
                cpd_possible = np.isclose(np.dot(charges, fractions), 0)
            else:
                oxidation_states = [self.data_source.get_oxidation_states(e) for e in elements]
                if self.fast:
                    # Assume each element can have only 1 oxidation state
                    cpd_possible = False
                    for ox in itertools.product(*oxidation_states):
                        if np.isclose(np.dot(ox, fractions), 0):
                            cpd_possible = True
                            break
                else:
                    #  Use pymatgen's oxidation state checker which
                    #   can detect whether an takes >1 oxidation state (as in Fe3O4)
                    oxi_state_dict = dict(zip([e.symbol for e in elements], oxidation_states))
                    cpd_possible = len(comp.oxi_state_guesses(oxi_states_override=oxi_state_dict)) > 0

            # Ionic character attributes
            atom_pairs = itertools.combinations(range(len(elements)), 2)
            el_frac = list(np.true_divide(fractions, sum(fractions)))

            ionic_char = []
            avg_ionic_char = 0

            for pair in atom_pairs:
                XA = elec[pair[0]]
                XB = elec[pair[1]]
                ionic_char.append(1.0 - np.exp(-0.25 * ((XA - XB) ** 2)))
                avg_ionic_char += el_frac[pair[0]] * el_frac[pair[1]] * ionic_char[-1]

            max_ionic_char = np.max(ionic_char)

        return [cpd_possible, max_ionic_char, avg_ionic_char]

    def feature_labels(self):
        labels = ["compound possible", "max ionic char", "avg ionic char"]
        return labels

    def citations(self):
        citation = [
            "@article{ward_agrawal_choudary_wolverton_2016, title={A general-purpose "
            "machine learning framework for predicting properties of inorganic materials}, "
            "volume={2}, DOI={10.1038/npjcompumats.2017.28}, number={1}, journal={npj "
            "Computational Materials}, author={Ward, Logan and Agrawal, Ankit and Choudhary, "
            "Alok and Wolverton, Christopher}, year={2016}}"
        ]
        return citation

    def implementors(self):
        return ["Jiming Chen", "Logan Ward"]


class ElectronAffinity(BaseFeaturizer):
    """
    Calculate average electron affinity times formal charge of anion elements.
    Note: The formal charges must already be computed before calling `featurize`.
    Generates average (electron affinity*formal charge) of anions.
    """

    def __init__(self):
        self.data_source = DemlData()

    def featurize(self, comp):
        """
        Args:
            comp: (Composition) Composition to be featurized

        Returns:
            avg_anion_affin (single-element list): average electron affinity*formal charge of anions
        """

        # Check if oxidation states have been computed
        if not has_oxidation_states(comp):
            raise ValueError("Composition lacks oxidation states")

        # Get the species and fractions
        species, fractions = zip(*comp.items())

        # Determine which species are anions
        anions, fractions = zip(*((s, f) for s, f in zip(species, fractions) if s.oxi_state < 0))

        # Compute the electron_affinity*formal_charge for each anion
        electron_affin = [
            self.data_source.get_elemental_property(s.element, "electron_affin") * s.oxi_state for s in anions
        ]

        # Compute the average affinity
        avg_anion_affin = PropertyStats.mean(electron_affin, fractions)

        return [avg_anion_affin]

    def feature_labels(self):
        return ["avg anion electron affinity"]

    def citations(self):
        citation = [
            "@article{deml_ohayre_wolverton_stevanovic_2016, title={Predicting density "
            "functional theory total energies and enthalpies of formation of metal-nonmetal "
            "compounds by linear regression}, volume={47}, DOI={10.1002/chin.201644254}, "
            "number={44}, journal={ChemInform}, author={Deml, Ann M. and Ohayre, Ryan and "
            "Wolverton, Chris and Stevanovic, Vladan}, year={2016}}"
        ]
        return citation

    def implementors(self):
        return ["Jiming Chen", "Logan Ward"]


class ElectronegativityDiff(BaseFeaturizer):
    """
    Features from electronegativity differences between anions and cations.

    These features are computed by first determining the concentration-weighted
    average electronegativity of the anions. For example, the average
    electronegativity of the anions in CaCoSO is equal to 1/2 of that of S and 1/2 of that of O.
    We then compute the difference between the electronegativity of each cation
    and the average anion electronegativity.

    The feature values are then determined based on the concentration-weighted statistics
    in the same manner as ElementProperty features. For example, one value could be
    the mean electronegativity difference over all the anions.

    Parameters:
        stats: Property statistics to compute

    Generates average electronegativity difference between cations and anions
    """

    def __init__(self, stats=None):
        if stats is None:
            self.stats = ["minimum", "maximum", "range", "mean", "std_dev"]
        else:
            self.stats = stats

    def featurize(self, comp):
        """
        Args:
            comp: Pymatgen Composition object

        Returns:
            en_diff_stats (list of floats): Property stats of electronegativity difference
        """

        # Check if oxidation states have been determined
        if not has_oxidation_states(comp):
            raise ValueError("Oxidation states have not yet been determined")
        if not is_ionic(comp):
            raise ValueError("Composition is not ionic")

        # Determine the average anion EN
        anions, anion_fractions = zip(*((s, x) for s, x in comp.items() if s.oxi_state < 0))

        # If there are no anions, raise an Exception
        if len(anions) == 0:
            raise Exception("Features not applicable: Compound contains no anions")

        anion_en = [s.element.X for s in anions]
        mean_anion_en = PropertyStats.mean(anion_en, anion_fractions)

        # Determine the EN difference for each cation
        cations, cation_fractions = zip(*((s, x) for s, x in comp.items() if s.oxi_state > 0))

        # If there are no cations, raise an Exception
        #  It is possible to construct a non-charge-balanced Composition,
        #    so we have to check for both the presence of anions and cations
        if len(cations) == 0:
            raise Exception("Features not applicable: Compound contains no cations")

        en_difference = [mean_anion_en - s.element.X for s in cations]

        # Compute the statistics
        return [PropertyStats.calc_stat(en_difference, stat, cation_fractions) for stat in self.stats]

    def feature_labels(self):
        labels = []
        for stat in self.stats:
            labels.append("%s EN difference" % stat)
        return labels

    def citations(self):
        citation = [
            "@article{deml_ohayre_wolverton_stevanovic_2016, title={Predicting density "
            "functional theory total energies and enthalpies of formation of metal-nonmetal "
            "compounds by linear regression}, volume={47}, DOI={10.1002/chin.201644254}, "
            "number={44}, journal={ChemInform}, author={Deml, Ann M. and Ohayre, Ryan and "
            "Wolverton, Chris and Stevanovic, Vladan}, year={2016}}"
        ]
        return citation

    def implementors(self):
        return ["Jiming Chen", "Logan Ward"]


def is_ionic(comp):
    """Determines whether a compound is an ionic compound.

    Looks at the oxidation states of each site and checks if both anions and cations exist

    Args:
        comp (Composition): Composition to check
    Returns:
        (bool) Whether the composition describes an ionic compound
    """

    has_cations = False
    has_anions = False

    for el in comp.elements:
        if el.oxi_state < 0:
            has_anions = True
        if el.oxi_state > 0:
            has_cations = True
        if has_anions and has_cations:
            return True
    return False
