from __future__ import division

import collections
import itertools
import os
from functools import reduce, lru_cache
from warnings import warn

import numpy as np
import pandas as pd
from pymatgen import Element, MPRester
from pymatgen.core.composition import Composition
from pymatgen.core.molecular_orbitals import MolecularOrbitals
from pymatgen.core.periodic_table import get_el_sp
from sklearn.neighbors.unsupervised import NearestNeighbors

from matminer.featurizers.base import BaseFeaturizer
from matminer.featurizers.utils.stats import PropertyStats
from matminer.utils.data import DemlData, MagpieData, PymatgenData, \
    CohesiveEnergyData, MixingEnthalpy

__author__ = 'Logan Ward, Jiming Chen, Ashwin Aggarwal, Kiran Mathew, ' \
             'Saurabh Bajaj, Qi Wang, Maxwell Dylla, Anubhav Jain'

module_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(module_dir, "..", "utils", "data_files")


# Utility operations
def has_oxidation_states(comp):
    """Check if a composition object has oxidation states for each element

    TODO: Does this make sense to add to pymatgen? -wardlt

    Args:
        comp - (Composition) Composition to check
    Returns:
        (Boolean) Whether this composition object contains oxidation states
    """
    for el in comp.elements:
        if not hasattr(el, "oxi_state") or el.oxi_state is None:
            return False
    return True


class ElementProperty(BaseFeaturizer):
    """
    Class to calculate elemental property attributes.

    To initialize quickly, use the from_preset() method.

    Args:
        data_source (AbstractData or str): source from which to retrieve
            element property data (or use str for preset: "pymatgen",
            "magpie", or "deml")
        features (list of strings): List of elemental properties to use
            (these must be supported by data_source)
        stats (list of strings): a list of weighted statistics to compute to for each
            property (see PropertyStats for available stats)
    """

    def __init__(self, data_source, features, stats):
        if data_source == "pymatgen":
            self.data_source = PymatgenData()
        elif data_source == "magpie":
            self.data_source = MagpieData()
        elif data_source == "deml":
            self.data_source = DemlData()
        else:
            self.data_source = data_source

        self.features = features
        self.stats = stats

    @classmethod
    def from_preset(cls, preset_name):
        """
        Return ElementProperty from a preset string
        Args:
            preset_name: (str) can be one of "magpie", "deml", or "matminer"

        Returns:

        """
        if preset_name == "magpie":
            data_source = "magpie"
            features = ["Number", "MendeleevNumber", "AtomicWeight",
                        "MeltingT",
                        "Column", "Row", "CovalentRadius",
                        "Electronegativity", "NsValence", "NpValence",
                        "NdValence", "NfValence", "NValence",
                        "NsUnfilled", "NpUnfilled", "NdUnfilled", "NfUnfilled",
                        "NUnfilled", "GSvolume_pa",
                        "GSbandgap", "GSmagmom", "SpaceGroupNumber"]
            stats = ["minimum", "maximum", "range", "mean", "avg_dev", "mode"]

        elif preset_name == "deml":
            data_source = "deml"
            stats = ["minimum", "maximum", "range", "mean", "std_dev"]
            features = ["atom_num", "atom_mass", "row_num", "col_num",
                        "atom_radius", "molar_vol", "heat_fusion",
                        "melting_point", "boiling_point", "heat_cap",
                        "first_ioniz", "electronegativity",
                        "electric_pol", "GGAU_Etot", "mus_fere",
                        "FERE correction"]

        elif preset_name == "matminer":
            data_source = "pymatgen"
            stats = ["minimum", "maximum", "range", "mean", "std_dev"]
            features = ["X", "row", "group", "block", "atomic_mass",
                        "atomic_radius", "mendeleev_no",
                        "electrical_resistivity", "velocity_of_sound",
                        "thermal_conductivity", "melting_point",
                        "bulk_modulus",
                        "coefficient_of_linear_thermal_expansion"]

        else:
            raise ValueError("Invalid preset_name specified!")

        return cls(data_source, features, stats)

    def featurize(self, comp):
        """
        Get elemental property attributes

        Args:
            comp: Pymatgen composition object

        Returns:
            all_attributes: Specified property statistics of features
        """

        all_attributes = []

        # Initialize stats computer
        pstats = PropertyStats()

        # Get the element names and fractions
        elements, fractions = zip(*comp.element_composition.items())

        for attr in self.features:
            elem_data = [self.data_source.get_elemental_property(e, attr) for e in elements]

            for stat in self.stats:
                all_attributes.append(pstats.calc_stat(elem_data, stat, fractions))

        return all_attributes

    def feature_labels(self):
        labels = []
        for attr in self.features:
            for stat in self.stats:
                labels.append("%s %s" % (stat, attr))
        return labels

    def citations(self):
        if self.data_source.__class__.__name__ == "MagpieData":
            citation = [
                "@article{ward_agrawal_choudary_wolverton_2016, title={A general-purpose "
                "machine learning framework for predicting properties of inorganic materials}, "
                "volume={2}, DOI={10.1038/npjcompumats.2017.28}, number={1}, journal={npj "
                "Computational Materials}, author={Ward, Logan and Agrawal, Ankit and Choudhary, "
                "Alok and Wolverton, Christopher}, year={2016}}"]
        elif self.data_source.__class__.__name__ == "DemlData":
            citation = [
                "@article{deml_ohayre_wolverton_stevanovic_2016, title={Predicting density "
                "functional theory total energies and enthalpies of formation of metal-nonmetal "
                "compounds by linear regression}, volume={47}, DOI={10.1002/chin.201644254}, "
                "number={44}, journal={ChemInform}, author={Deml, Ann M. and Ohayre, Ryan and "
                "Wolverton, Chris and Stevanovic, Vladan}, year={2016}}"]
        elif self.data_source.__class__.__name__ == "PymatgenData":
            citation = [
                "@article{Ong2013, author = {Ong, Shyue Ping and Richards, William Davidson and Jain, Anubhav and Hautier, "
                "Geoffroy and Kocher, Michael and Cholia, Shreyas and Gunter, Dan and Chevrier, Vincent L. and Persson, "
                "Kristin A. and Ceder, Gerbrand}, doi = {10.1016/j.commatsci.2012.10.028}, issn = {09270256}, "
                "journal = {Computational Materials Science}, month = {feb}, pages = {314--319}, "
                "publisher = {Elsevier B.V.}, title = {{Python Materials Genomics (pymatgen): A robust, open-source python "
                "library for materials analysis}}, url = {http://linkinghub.elsevier.com/retrieve/pii/S0927025612006295}, "
                "volume = {68}, year = {2013} } "]
        else:
            citation = []
        return citation

    def implementors(self):
        return ["Jiming Chen", "Logan Ward", "Anubhav Jain"]


class CationProperty(ElementProperty):
    """
    Features based on properties of cations in a material

    Requires that oxidation states have already been determined

    Computes composition-weighted statistics of different elemental properties
    """

    @classmethod
    def from_preset(cls, preset_name):
        if preset_name == "deml":
            data_source = "deml"
            features = ["total_ioniz", "xtal_field_split", "magn_moment", "so_coupling", "sat_magn"]
            stats = ["minimum", "maximum", "range", "mean", "std_dev"]
        else:
            raise ValueError('Preset "%s" not found'%preset_name)
        return cls(data_source, features, stats)

    def feature_labels(self):
        labels = []
        for attr in self.features:
            for stat in self.stats:
                labels.append("%s %s of cations"%(stat, attr))

        return labels

    def featurize(self, comp):
        # Check if oxidation states are present
        if not has_oxidation_states(comp):
            raise ValueError('Oxidation states have not been determined')

        # Prepare to store the attributes
        all_attributes = []

        # Initialize stats computer
        pstats = PropertyStats()

        # Get the cation species and fractions
        cations, fractions = zip(*[(s, f) for s, f in comp.items() if s.oxi_state > 0])

        for attr in self.features:
            elem_data = [self.data_source.get_charge_dependent_property_from_specie(c, attr)
                         for c in cations]

            for stat in self.stats:
                all_attributes.append(pstats.calc_stat(elem_data, stat, fractions))

        return all_attributes

    def citations(self):
        return [
            "@article{deml_ohayre_wolverton_stevanovic_2016, title={Predicting density "
            "functional theory total energies and enthalpies of formation of metal-nonmetal "
            "compounds by linear regression}, volume={47}, DOI={10.1002/chin.201644254}, "
            "number={44}, journal={ChemInform}, author={Deml, Ann M. and Ohayre, Ryan and "
            "Wolverton, Chris and Stevanovic, Vladan}, year={2016}}"]


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
            raise ValueError('Oxidation states have not been determined')

        # Get the oxidation states and their proportions
        oxid_states, fractions = zip(*[(s.oxi_state, f) for s, f in comp.items()])

        # Compute statistics
        return [PropertyStats.calc_stat(oxid_states, s, fractions) for s in self.stats]

    def feature_labels(self):
        return ["%s oxidation state"%s for s in self.stats]

    def citations(self):
        return ["@article{deml_ohayre_wolverton_stevanovic_2016, title={Predicting density "
            "functional theory total energies and enthalpies of formation of metal-nonmetal "
            "compounds by linear regression}, volume={47}, DOI={10.1002/chin.201644254}, "
            "number={44}, journal={ChemInform}, author={Deml, Ann M. and Ohayre, Ryan and "
            "Wolverton, Chris and Stevanovic, Vladan}, year={2016}}"]

    def implementors(self):
        return ['Logan Ward']


class AtomicOrbitals(BaseFeaturizer):
    """
    Determine HOMO/LUMO features based on a composition.

    The highest occupied molecular orbital (HOMO) and lowest unoccupied
    molecular orbital (LUMO) are estiated from the atomic orbital energies
    of the composition. The atomic orbital energies are from NIST:
    https://www.nist.gov/pml/data/atomic-reference-data-electronic-structure-calculations

    Warning:
    For compositions with inter-species fractions greater than 10,000 (e.g.
    dilute alloys such as FeC0.00001) the composition will be truncated (to Fe
    in this example). In such extreme cases, the truncation likely reflects the
    true physics of the situation (i.e. that the dilute element does not
    significantly contribute orbital character to the band structure), but the
    user should be aware of this behavior.
    """

    def featurize(self, comp):
        """
        Args:
            comp: (Composition)
                pymatgen Composition object

        Returns:
            HOMO_character: (str) orbital symbol ('s', 'p', 'd', or 'f')
            HOMO_element: (str) symbol of element for HOMO
            HOMO_energy: (float in eV) absolute energy of HOMO
            LUMO_character: (str) orbital symbol ('s', 'p', 'd', or 'f')
            LUMO_element: (str) symbol of element for LUMO
            LUMO_energy: (float in eV) absolute energy of LUMO
            gap_AO: (float in eV)
                the estimated bandgap from HOMO and LUMO energeis
        """

        integer_comp, factor = comp.get_integer_formula_and_factor()

        # warning message if composition is dilute and truncated
        if not (len(Composition(comp).elements) ==
                len(Composition(integer_comp).elements)):
            warn('AtomicOrbitals: {} truncated to {}'.format(comp,
                                                             integer_comp))

        homo_lumo = MolecularOrbitals(integer_comp).band_edges

        feat = collections.OrderedDict()
        for edge in ['HOMO', 'LUMO']:
            feat['{}_character'.format(edge)] = homo_lumo[edge][1][-1]
            feat['{}_element'.format(edge)] = homo_lumo[edge][0]
            feat['{}_energy'.format(edge)] = homo_lumo[edge][2]
        feat['gap_AO'] = feat['LUMO_energy'] - feat['HOMO_energy']

        return list(feat.values())

    def feature_labels(self):
        feat = []
        for edge in ['HOMO', 'LUMO']:
            feat.extend(['{}_character'.format(edge),
                         '{}_element'.format(edge),
                         '{}_energy'.format(edge)])
        feat.append("gap_AO")
        return feat

    def citations(self):
        return [
            "@article{PhysRevA.55.191,"
            "title = {Local-density-functional calculations of the energy of atoms},"
            "author = {Kotochigova, Svetlana and Levine, Zachary H. and Shirley, "
            "Eric L. and Stiles, M. D. and Clark, Charles W.},"
            "journal = {Phys. Rev. A}, volume = {55}, issue = {1}, pages = {191--199},"
            "year = {1997}, month = {Jan}, publisher = {American Physical Society},"
            "doi = {10.1103/PhysRevA.55.191}, "
            "url = {https://link.aps.org/doi/10.1103/PhysRevA.55.191}}"]

    def implementors(self):
        return ['Maxwell Dylla', 'Anubhav Jain']


class BandCenter(BaseFeaturizer):
    """
    Estimation of absolute position of band center using electronegativity.

    Features
        - Band center
    """

    def featurize(self, comp):
        """
        (Rough) estimation of absolution position of band center using
        geometric mean of electronegativity.

        Args:
            comp (Composition).

        Returns:
            (float) band center.

        """
        prod = 1.0
        for el, amt in comp.get_el_amt_dict().items():
            prod = prod * (Element(el).X ** amt)

        return [-prod ** (1 / sum(comp.get_el_amt_dict().values()))]

    def feature_labels(self):
        return ["band center"]

    def citations(self):
        return [
            "@article{Butler1978, author = {Butler, M A and Ginley, D S}, "
            "doi = {10.1149/1.2131419}, isbn = {0013-4651}, issn = {00134651}, "
            "journal = {Journal of The Electrochemical Society}, month = {feb},"
            " number = {2}, pages = {228--232}, title = {{Prediction of "
            "Flatband Potentials at Semiconductor-Electrolyte Interfaces from "
            "Atomic Electronegativities}}, url = "
            "{http://jes.ecsdl.org/content/125/2/228}, volume = {125}, "
            "year = {1978} } "]

    def implementors(self):
        return ["Anubhav Jain"]


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
        data_source (data class): source from which to retrieve element data
        stats: Property statistics to compute

    Generates average electronegativity difference between cations and anions
    """

    def __init__(self, stats=None):
        if stats == None:
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
            raise ValueError('Oxidation states have not yet been determined')

        # Determine the average anion EN
        anions, anion_fractions = zip(*[(s, x) for s, x in comp.items() if s.oxi_state < 0])

        # If there are no anions, raise an Exception
        if len(anions) == 0:
            raise Exception('Features not applicable: Compound contains no anions')

        anion_en = [s.element.X for s in anions]
        mean_anion_en = PropertyStats.mean(anion_en, anion_fractions)

        # Determine the EN difference for each cation
        cations, cation_fractions = zip(*[(s, x) for s, x in comp.items() if s.oxi_state > 0])

        # If there are no cations, raise an Exception
        #  It is possible to construct a non-charge-balanced Composition,
        #    so we have to check for both the presence of anions and cations
        if len(cations) == 0:
            raise Exception('Features not applicable: Compound contains no cations')

        en_difference = [mean_anion_en - s.element.X for s in cations]

        # Compute the statistics
        return [
            PropertyStats.calc_stat(en_difference, stat, cation_fractions) for stat in self.stats
        ]

    def feature_labels(self):
        labels = []
        for stat in self.stats:
            labels.append("%s EN difference" % stat)
        return labels

    def citations(self):
        citation = ["@article{deml_ohayre_wolverton_stevanovic_2016, title={Predicting density "
            "functional theory total energies and enthalpies of formation of metal-nonmetal "
            "compounds by linear regression}, volume={47}, DOI={10.1002/chin.201644254}, "
            "number={44}, journal={ChemInform}, author={Deml, Ann M. and Ohayre, Ryan and "
            "Wolverton, Chris and Stevanovic, Vladan}, year={2016}}"]
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
            raise ValueError('Composition lacks oxidation states')

        # Get the species and fractions
        species, fractions = zip(*comp.items())

        # Determine which species are anions
        anions, fractions = zip(*[(s, f) for s, f in zip(species, fractions) if s.oxi_state < 0])

        # Compute the electron_affinity*formal_charge for each anion
        electron_affin = [
            self.data_source.get_elemental_property(s.element, "electron_affin") * s.oxi_state
            for s in anions
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
            "Wolverton, Chris and Stevanovic, Vladan}, year={2016}}"]
        return citation

    def implementors(self):
        return ["Jiming Chen", "Logan Ward"]


class Stoichiometry(BaseFeaturizer):
    """
    Calculate norms of stoichiometric attributes.

    Parameters:
        p_list (list of ints): list of norms to calculate
        num_atoms (bool): whether to return number of atoms per formula unit
    """

    def __init__(self, p_list=(0, 2, 3, 5, 7, 10), num_atoms=False):
        self.p_list = p_list
        self.num_atoms = num_atoms

    def featurize(self, comp):
        """
        Get stoichiometric attributes
        Args:
            comp: Pymatgen composition object
            p_list (list of ints)

        Returns:
            p_norm (list of floats): Lp norm-based stoichiometric attributes.
                Returns number of atoms if no p-values specified.
        """

        el_amt = comp.get_el_amt_dict()

        # Compute the number of atoms per formula unit
        n_atoms_per_unit = comp.num_atoms / \
                           comp.get_integer_formula_and_factor()[1]

        if self.p_list is None:
            stoich_attr = [n_atoms_per_unit]  # return num atoms if no norms specified
        else:
            p_norms = [0] * len(self.p_list)
            n_atoms = sum(el_amt.values())

            for i in range(len(self.p_list)):
                if self.p_list[i] < 0:
                    raise ValueError("p-norm not defined for p < 0")
                if self.p_list[i] == 0:
                    p_norms[i] = len(el_amt.values())
                else:
                    for j in el_amt:
                        p_norms[i] += (el_amt[j] / n_atoms) ** self.p_list[i]
                    p_norms[i] = p_norms[i] ** (1.0 / self.p_list[i])

            if self.num_atoms:
                stoich_attr = [n_atoms_per_unit] + p_norms
            else:
                stoich_attr = p_norms

        return stoich_attr

    def feature_labels(self):
        labels = []
        if self.num_atoms:
            labels.append("num atoms")

        if self.p_list != None:
            for p in self.p_list:
                labels.append("%d-norm" % p)

        return labels

    def citations(self):
        citation = [
            "@article{ward_agrawal_choudary_wolverton_2016, title={A general-purpose "
            "machine learning framework for predicting properties of inorganic materials}, "
            "volume={2}, DOI={10.1038/npjcompumats.2017.28}, number={1}, journal={npj "
            "Computational Materials}, author={Ward, Logan and Agrawal, Ankit and Choudhary, "
            "Alok and Wolverton, Christopher}, year={2016}}"]
        return citation

    def implementors(self):
        return ["Jiming Chen", "Logan Ward"]


class ValenceOrbital(BaseFeaturizer):
    """
        Attributes of valence orbital shells

        Args:
            data_source (data object): source from which to retrieve element data
            orbitals (list): orbitals to calculate
            props (list): specifies whether to return average number of electrons in each orbital,
                fraction of electrons in each orbital, or both
    """

    def __init__(self, orbitals=("s", "p", "d", "f"), props=("avg", "frac")):
        self.data_source = MagpieData()
        self.orbitals = orbitals
        self.props = props

    def featurize(self, comp):
        """Weighted fraction of valence electrons in each orbital

           Args:
                comp: Pymatgen composition object

           Returns:
                valence_attributes (list of floats): Average number and/or
                    fraction of valence electrons in specfied orbitals
        """

        elements, fractions = zip(*comp.element_composition.items())

        # Get the mean number of electrons in each shell
        avg = [
            PropertyStats.mean(
                self.data_source.get_elemental_properties(elements, "N%sValence" % orb),
                weights=fractions)
            for orb in self.orbitals
        ]

        # If needed, get fraction of electrons in each shell
        if "frac" in self.props:
            avg_total_valence = PropertyStats.mean(
                self.data_source.get_elemental_properties(elements, "NValence"),
                weights=fractions)
            frac = [a / avg_total_valence for a in avg]

        # Get the desired attributes
        valence_attributes = []
        for prop in self.props:
            valence_attributes += locals()[prop]

        return valence_attributes

    def feature_labels(self):
        labels = []
        for prop in self.props:
            for orb in self.orbitals:
                labels.append("%s %s valence electrons" % (prop, orb))

        return labels

    def citations(self):
        ward_citation = (
            "@article{ward_agrawal_choudary_wolverton_2016, title={A general-purpose "
            "machine learning framework for predicting properties of inorganic materials}, "
            "volume={2}, DOI={10.1038/npjcompumats.2017.28}, number={1}, journal={npj "
            "Computational Materials}, author={Ward, Logan and Agrawal, Ankit and Choudhary, "
            "Alok and Wolverton, Christopher}, year={2016}}")
        deml_citation = (
            "@article{deml_ohayre_wolverton_stevanovic_2016, title={Predicting density "
            "functional theory total energies and enthalpies of formation of metal-nonmetal "
            "compounds by linear regression}, volume={47}, DOI={10.1002/chin.201644254}, "
            "number={44}, journal={ChemInform}, author={Deml, Ann M. and Ohayre, Ryan and "
            "Wolverton, Chris and Stevanovic, Vladan}, year={2016}}")
        citations = [ward_citation, deml_citation]
        return citations

    def implementors(self):
        return ["Jiming Chen", "Logan Ward"]


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
                charges, fractions = zip(*[(s.oxi_state, f) for s, f in comp.items()])
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
                    oxi_state_dict = dict(zip([e.symbol for e in elements],
                                              oxidation_states))
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
                avg_ionic_char += el_frac[pair[0]] * el_frac[pair[1]] * \
                                  ionic_char[-1]

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
            "Alok and Wolverton, Christopher}, year={2016}}"]
        return citation

    def implementors(self):
        return ["Jiming Chen", "Logan Ward"]


class ElementFraction(BaseFeaturizer):
    """
    Class to calculate the atomic fraction of each element in a composition.

    Generates a vector where each index represents an element in atomic number order.
    """

    def __init__(self):
        pass

    def featurize(self, comp):
        """
        Args:
            comp: Pymatgen Composition object

        Returns:
            vector (list of floats): fraction of each element in a composition
        """

        vector = [0] * 103
        el_list = list(comp.element_composition.fractional_composition.items())
        for el in el_list:
            obj = el
            atomic_number_i = obj[0].number - 1
            vector[atomic_number_i] = obj[1]
        return vector

    def feature_labels(self):
        labels = []
        for i in range(1, 104):
            labels.append(Element.from_Z(i).symbol)
        return labels

    def implementors(self):
        return ["Ashwin Aggarwal", "Logan Ward"]

    def citations(self):
        return []


class TMetalFraction(BaseFeaturizer):
    """
    Class to calculate fraction of magnetic transition metals in a composition.

    Parameters:
        data_source (data class): source from which to retrieve element data

    Generates: Fraction of magnetic transition metal atoms in a compound
    """

    def __init__(self):
        self.magn_elem = ['Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Nb',
                          'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Ta', 'W', 'Re',
                          'Os', 'Ir', 'Pt']

    def featurize(self, comp):
        """
        Args:
            comp: Pymatgen Composition object

        Returns:
            frac_magn_atoms (single-element list): fraction of magnetic transitional metal atoms in a compound
        """

        el_amt = comp.get_el_amt_dict()

        frac_magn_atoms = 0
        for el in el_amt:
            if el in self.magn_elem:
                frac_magn_atoms += el_amt[el]

        frac_magn_atoms /= sum(el_amt.values())

        return [frac_magn_atoms]

    def feature_labels(self):
        labels = ["transition metal fraction"]
        return labels

    def citations(self):
        citation = [
            "@article{deml_ohayre_wolverton_stevanovic_2016, title={Predicting density "
            "functional theory total energies and enthalpies of formation of metal-nonmetal "
            "compounds by linear regression}, volume={47}, DOI={10.1002/chin.201644254}, "
            "number={44}, journal={ChemInform}, author={Deml, Ann M. and Ohayre, Ryan and "
            "Wolverton, Chris and Stevanovic, Vladan}, year={2016}}"]
        return citation

    def implementors(self):
        return ["Jiming Chen, Logan Ward"]


class CohesiveEnergy(BaseFeaturizer):
    """
    Cohesive energy per atom using elemental cohesive energies and
    formation energy.

    Get cohesive energy per atom of a compound by adding known
    elemental cohesive energies from the formation energy of the
    compound.

    Parameters:
        mapi_key (str): Materials API key for looking up formation energy
            by composition alone (if you don't set the formation energy
            yourself).
    """

    def __init__(self, mapi_key=None):
        self.mapi_key = mapi_key

    def featurize(self, comp, formation_energy_per_atom=None):
        """

        Args:
            comp: (str) compound composition, eg: "NaCl"
            formation_energy_per_atom: (float) the formation energy per atom of
                your compound. If not set, will look up the most stable
                formation energy from the Materials Project database.
        """
        el_amt_dict = comp.get_el_amt_dict()

        formation_energy_per_atom = formation_energy_per_atom or None

        if not formation_energy_per_atom:
            # Get formation energy of most stable structure from MP
            struct_lst = MPRester(self.mapi_key).get_data(
                comp.formula.replace(" ", ""))
            if len(struct_lst) > 0:
                most_stable_entry = sorted(struct_lst, key=lambda e: e['energy_per_atom'])[0]
                formation_energy_per_atom = most_stable_entry[
                    'formation_energy_per_atom']
            else:
                raise ValueError('No structure found in MP for {}'.format(comp))

        # Subtract elemental cohesive energies from formation energy
        cohesive_energy = -formation_energy_per_atom * comp.num_atoms
        for el in el_amt_dict:
            cohesive_energy += el_amt_dict[el] * \
                               CohesiveEnergyData().get_elemental_property(el)

        cohesive_energy_per_atom = cohesive_energy / comp.num_atoms

        return [cohesive_energy_per_atom]

    def feature_labels(self):
        return ["cohesive energy"]

    def implementors(self):
        return ["Saurabh Bajaj", "Anubhav Jain"]

    def citations(self):
        # Cohesive energy values for the elements are taken from the
        # Knowledgedoor web site, which obtained those values from Kittel.
        # We include both citations.
        return [
            "@misc{, title = {{Knowledgedoor Cohesive energy handbook}}, "
            "url = {http://www.knowledgedoor.com/2/elements{\_}handbook/cohesive{\_}energy.html}}",
            "@book{Kittel, author = {Kittel, C}, isbn = {978-0-471-41526-8}, "
            "publisher = {Wiley}, title = {{Introduction to Solid State "
            "Physics, 8th Edition}}, year = {2005}}"]


class Miedema(BaseFeaturizer):
    """
    Formation enthalpies of intermetallic compounds, from Miedema et al.

    Calculate the formation enthalpies of the intermetallic compound,
    solid solution and amorphous phase of a given composition, based on
    semi-empirical Miedema model (and some extensions), particularly for
    transitional metal alloys.
    Support elemental, binary and multicomponent alloys.
        For elemental/binary alloys, the formulation is based on the original
        works by Miedema et al. in 1980s;
        For multicomponent alloys, the formulation is basically the linear
        combination of sub-binary systems. This is reported to work well for
        ternary alloys, but needs to be careful with quaternary alloys and more.

    Args:
        struct_types (str or list of str): default='inter'
            if str, one target structure;
            if list, a list of target structures.
            e.g.
            'inter': intermetallic compound
            'ss': solid solution
            'amor': amorphous phase
            'all': same for ['inter', 'ss', 'amor']
            ['inter', 'ss']: amorphous phase and solid solution, as an example
        ss_types (str or list of str): only for ss, default='min'
            if str, one structure type of ss;
            if list, a list of structure types of ss.
            e.g.
            'fcc': fcc solid solution
            'bcc': bcc solid solution
            'hcp': hcp solid solution
            'no_latt': solid solution with no specific structure type
            'min': min value of ['fcc', 'bcc', 'hcp', 'no_latt']
            'all': same for ['fcc', 'bcc', 'hcp', 'no_latt']
            ['fcc', 'bcc']: fcc and bcc solid solutions, as an example
        data_source (str): default='Miedema', source of dataset
            'Miedema': read from 'Miedema.csv'
                        parameterized by Miedema et al. in 1980s,
                        containing parameters for 73 types of elements:
                         'molar_volume'
                         'electron_density'
                         'electronegativity'
                         'valence_electrons'
                         'a_const'
                         'R_const'
                         'H_trans'
                         'compressibility'
                         'shear_modulus'
                         'melting_point'
                         'structural_stability'
    Returns:
        (list of floats) Miedema formation enthalpies (per atom)
            -formation_enthalpy_inter: for intermetallic compound
            -formation_enthalpy_ss: for solid solution, can be divided into
                                   'min', 'fcc', 'bcc', 'hcp', 'no_latt'
                                    for different lattice_types
            -formation_enthalpy_amor: for amorphous phase
    """

    def __init__(self, struct_types='inter', ss_types='min',
                 data_source='Miedema'):
        if isinstance(struct_types, list):
            self.struct_types = struct_types
        else:
            if struct_types == 'all':
                self.struct_types = ['inter', 'amor', 'ss']
            else:
                self.struct_types = [struct_types]

        if isinstance(ss_types, list):
            self.ss_types = ss_types
        else:
            if ss_types == 'all':
                self.ss_types = ['fcc', 'bcc', 'hcp', 'no_latt']
            else:
                self.ss_types = [ss_types]

        self.data_source = data_source
        if self.data_source == 'Miedema':
            self.df_dataset = pd.read_csv(
                os.path.join(data_dir, 'Miedema.csv'), index_col='element')
        else:
            raise NotImplementedError('data_source {} not implemented yet'.
                                      format(self, data_source))

    def deltaH_chem(self, elements, fracs, struct):
        """
        Chemical term of formation enthalpy
        Args:
            elements (list of str): list of elements
            fracs (list of floats): list of atomic fractions
            struct (str): 'inter', 'ss' or 'amor'
        Returns:
            deltaH_chem (float): chemical term of formation enthalpy
        """
        if any([el not in self.df_dataset.index for el in elements]):
            return np.nan
        df_el = self.df_dataset.loc[elements]
        v_molar = np.array(df_el['molar_volume'])
        n_ws = np.array(df_el['electron_density'])
        elec = np.array(df_el['electronegativity'])
        val = np.array(df_el['valence_electrons'])
        a = np.array(df_el['a_const'])
        r = np.array(df_el['R_const'])
        h_trans = np.array(df_el['H_trans'])

        if struct == 'inter':
            gamma = 8
        elif struct == 'amor':
            gamma = 5
        else:
            gamma = 0

        c_sf = (fracs * np.power(v_molar, 2 / 3) / np.dot(fracs, np.power(v_molar, 2 / 3)))
        f = (c_sf * (1 + gamma * np.power(np.multiply.reduce(c_sf, 0), 2)))[::-1]
        v_a = np.array([np.power(v_molar[0], 2 / 3) * (1 + a[0] * f[0] * (elec[0] - elec[1])),
                        np.power(v_molar[1], 2 / 3) * (1 + a[1] * f[1] * (elec[1] - elec[0]))])
        c_sf_a = fracs * v_a / np.dot(fracs, v_a)
        f_a = (c_sf_a * (1 + gamma * np.power(np.multiply.reduce
                                              (c_sf_a, 0), 2)))[::-1]

        threshold = range(3, 12)
        if (val[0] in threshold) and (val[1] in threshold):
            p = 14.1
            r = 0.
        elif (val[0] not in threshold) and (val[1] not in threshold):
            p = 10.7
            r = 0.
        else:
            p = 12.35
            r = np.multiply.reduce(r, 0) * p
        q = p * 9.4

        eta_ab = (2 * (-p * np.power(elec[0] - elec[1], 2) - r +
                       q * np.power(np.power(n_ws[0], 1 / 3) -
                                    np.power(n_ws[1], 1 / 3), 2)) /
                  reduce(lambda x, y: 1 / x + 1 / y, np.power(n_ws, 1 / 3)))

        deltaH_chem = (f_a[0] * fracs[0] * v_a[0] * eta_ab +
                       np.dot(fracs, h_trans))
        return deltaH_chem

    def deltaH_elast(self, elements, fracs):
        """
        Elastic term of formation enthalpy
        Args:
            elements (list of str): list of elements
            fracs (list of floats): list of atomic fractions
        Returns:
            deltaH_elastic (float): elastic term of formation enthalpy
        """
        if any([el not in self.df_dataset.index for el in elements]):
            return np.nan
        df_el = self.df_dataset.loc[elements]
        v_molar = np.array(df_el['molar_volume'])
        n_ws = np.array(df_el['electron_density'])
        elec = np.array(df_el['electronegativity'])
        compr = np.array(df_el['compressibility'])
        shear_mod = np.array(df_el['shear_modulus'])

        alp = (np.multiply(1.5, np.power(v_molar, 2 / 3)) /
               reduce(lambda x, y: 1 / x + 1 / y, np.power(n_ws, 1 / 3)))
        v_a = (v_molar + np.array([alp[0] * (elec[0] - elec[1]) / n_ws[0],
                                   alp[1] * (elec[1] - elec[0]) / n_ws[1]]))
        alp_a = (np.multiply(1.5, np.power(v_a, 2 / 3)) /
                 reduce(lambda x, y: 1 / x + 1 / y, np.power(n_ws, 1 / 3)))

        # effective volume in alloy
        vab_a = (v_molar[0] +
                 np.array([alp_a[0] * (elec[0] - elec[1]) / n_ws[0],
                           alp_a[1] * (elec[1] - elec[0]) / n_ws[0]]))
        vba_a = (v_molar[1] +
                 np.array([alp_a[0] * (elec[0] - elec[1]) / n_ws[1],
                           alp_a[1] * (elec[1] - elec[0]) / n_ws[1]]))

        # H_elast A in B
        hab_elast = ((2 * compr[0] * shear_mod[1] *
                      np.power((vab_a[0] - vba_a[0]), 2)) /
                     (4 * shear_mod[1] * vab_a[0] +
                      3 * compr[0] * vba_a[0]))
        # H_elast B in A
        hba_elast = ((2 * compr[1] * shear_mod[0] *
                      np.power((vba_a[1] - vab_a[1]), 2)) /
                     (4 * shear_mod[0] * vba_a[1] +
                      3 * compr[1] * vab_a[1]))

        deltaH_elast = (np.multiply.reduce(fracs, 0) *
                        (fracs[1] * hab_elast + fracs[0] * hba_elast))
        return deltaH_elast

    def deltaH_struct(self, elements, fracs, latt):
        """
        Structural term of formation enthalpy, only for solid solution
        Args:
            elements (list of str): list of elements
            fracs (list of floats): list of atomic fractions
            latt (str): 'fcc', 'bcc', 'hcp' or 'no_latt'
        Returns:
            deltaH_struct (float): structural term of formation enthalpy
        """
        if any([el not in self.df_dataset.index for el in elements]):
            return np.nan
        df_el = self.df_dataset.loc[elements]
        val = np.array(df_el['valence_electrons'])
        struct_stab = np.array(df_el['structural_stability'])

        if latt == 'fcc':
            latt_stab_dict = {0.: 0., 1.: 0, 2.: 0, 3.: -2, 4.: -1.5,
                              5.: 9., 5.5: 14., 6.: 11., 7.: -3., 8.: -9.5,
                              8.5: -11., 9.: -9., 10.: -2., 11.: 1.5,
                              12.: 0., 13.: 0., 14.: 0., 15.: 0.}
        elif latt == 'bcc':
            latt_stab_dict = {0.: 0., 1.: 0., 2.: 0., 3.: 2.2, 4.: 2.,
                              5.: -9.5, 5.5: -14.5, 6.: -12., 7.: 4.,
                              8.: 10., 8.5: 11., 9.: 8.5, 10.: 1.5,
                              11.: 1.5, 12.: 0., 13.: 0., 14.: 0., 15.: 0.}
        elif latt == 'hcp':
            latt_stab_dict = {0.: 0., 1.: 0., 2.: 0., 3.: -2.5, 4.: -2.5,
                              5.: 10., 5.5: 15., 6.: 13., 7.: -5.,
                              8.: -10.5, 8.5: -11., 9.: -8., 10.: -1.,
                              11.: 2.5, 12.: 0., 13.: 0., 14.: 0., 15.: 0.}
        else:
            return 0
        latt_stab_dict = collections.OrderedDict(sorted(latt_stab_dict.items(),
                                                        key=lambda t: t[0]))
        # lattice stability of different lattice_types
        val_avg = np.dot(fracs, val)
        val_bd_lower, val_bd_upper = 0, 0
        for key in latt_stab_dict.keys():
            if val_avg - key <= 0:
                val_bd_upper = key
                break
            else:
                val_bd_lower = key

        latt_stab = ((val_avg - val_bd_lower) * latt_stab_dict[val_bd_upper] /
                     (val_bd_upper - val_bd_lower) +
                     (val_bd_upper - val_avg) * latt_stab_dict[val_bd_lower] /
                     (val_bd_upper - val_bd_lower))

        deltaH_struct = latt_stab - np.dot(fracs, struct_stab)
        return deltaH_struct

    def deltaH_topo(self, elements, fracs):
        """
        Topological term of formation enthalpy, only for amorphous phase
        Args:
            elements (list of str): list of elements
            fracs (list of floats): list of atomic fractions
        Returns:
            deltaH_topo (float): topological term of formation enthalpy
        """
        if any([el not in self.df_dataset.index for el in elements]):
            return np.nan
        df_el = self.df_dataset.loc[elements]
        melt_point = np.array(df_el['melting_point'])

        deltaH_topo = 3.5 * np.dot(fracs, melt_point) / 1000
        return deltaH_topo

    def featurize(self, comp):
        """
        Get Miedema formation enthalpies of target structures: inter, amor,
        ss (can be further divided into 'min', 'fcc', 'bcc', 'hcp', 'no_latt'
            for different lattice_types)
        Args:
            comp: Pymatgen composition object
        Returns:
            miedema (list of floats): formation enthalpies of target structures
        """
        el_amt = comp.fractional_composition.get_el_amt_dict()
        elements = sorted(el_amt.keys(), key=lambda sym: get_el_sp(sym).X)
        fracs = [el_amt[el] for el in elements]
        el_num = len(elements)
        # divide into a list of sub-binaries
        el_bins = []
        frac_bins = []
        for i in range(el_num - 1):
            for j in range(i + 1, el_num):
                el_bins.append([elements[i], elements[j]])
                frac_bins.append([fracs[i], fracs[j]])

        miedema = []
        for struct_type in self.struct_types:
            # inter: intermetallic compound
            if struct_type == 'inter':
                deltaH_chem_inter = 0
                for i_inter, el_bin in enumerate(el_bins):
                    deltaH_chem_inter += self.deltaH_chem(el_bin,
                                                          frac_bins[i_inter],
                                                          'inter')
                miedema.append(deltaH_chem_inter)
            # ss: solid solution
            elif struct_type == 'ss':
                deltaH_chem_ss = 0
                deltaH_elast_ss = 0
                for sub_bin, el_bin in enumerate(el_bins):
                    deltaH_chem_ss += self.deltaH_chem(el_bin, frac_bins[sub_bin], 'ss')
                    deltaH_elast_ss += self.deltaH_elast(el_bin, frac_bins[sub_bin])

                for ss_type in self.ss_types:
                    if ss_type == 'min':
                        deltaH_ss_all = []
                        for latt in ['fcc', 'bcc', 'hcp', 'no_latt']:
                            deltaH_ss_all.append(
                                deltaH_chem_ss + deltaH_elast_ss +
                                self.deltaH_struct(elements, fracs, latt))
                        deltaH_ss_min = min(deltaH_ss_all)
                        miedema.append(deltaH_ss_min)
                    else:
                        deltaH_struct_ss = self.deltaH_struct(elements,
                                                              fracs, ss_type)
                        miedema.append(deltaH_chem_ss + deltaH_elast_ss +
                                       deltaH_struct_ss)
            # amor: amorphous phase
            elif struct_type == 'amor':
                deltaH_chem_amor = 0
                deltaH_topo_amor = self.deltaH_topo(elements, fracs)
                for sub_bin, el_bin in enumerate(el_bins):
                    deltaH_chem_amor += self.deltaH_chem(el_bin,
                                                         frac_bins[sub_bin],
                                                         'amor')
                miedema.append(deltaH_chem_amor + deltaH_topo_amor)

        # convert kJ/mol to eV/atom. The original Miedema model is in kJ/mol.
        miedema = [deltaH / 96.4853 for deltaH in miedema]
        return miedema

    def feature_labels(self):
        labels = []
        for struct_type in self.struct_types:
            if struct_type == 'ss':
                for ss_type in self.ss_types:
                    labels.append('Miedema_deltaH_ss_' + ss_type)
            else:
                labels.append('Miedema_deltaH_' + struct_type)
        return labels

    def citations(self):
        miedema_citation = (
            '@article{miedema_1988, '
            'title={Cohesion in metals},'
            'author={De Boer, Frank R and Mattens, WCM '
            'and Boom, R and Miedema, AR and Niessen, AK},'
            'year={1988}}')
        zhang_citation = (
            '@article{miedema_zhang_2016, '
            'title={Miedema Calculator: A thermodynamic platform '
            'for predicting formation enthalpies of alloys within '
            'framework of Miedema\'s Theory},'
            'author={R.F. Zhang, S.H. Zhang, Z.J. He, J. Jing and S.H. Sheng},'
            'journal={Computer Physics Communications}'
            'year={2016}}')
        ternary_citation = (
            '@article{miedema_alonso_1990, '
            'title={Glass formation in ternary transition metal alloys},'
            'author={L J Gallego, J A Somoza and J A Alonso},'
            'journal={Journal of Physics: Condensed Matter}'
            'year={1990}}')
        return [miedema_citation, zhang_citation, ternary_citation]

    def implementors(self):
        return ['Qi Wang', 'Alireza Faghaninia']


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

    def featurize(self, comp):
        return [self.compute_omega(comp), self.compute_delta(comp)]

    def compute_omega(self, comp):
        """Compute Yang's mixing thermodynamics descriptor

        :math:`\frac{T_m \Delta S_{mix}}{ |  \Delta H_{mix} | }`

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

        return abs(mean_Tm * entropy / enthalpy)

    def compute_delta(self, comp):
        """Compute Yang's delta parameter

        :math:`\sqrt{\sum^n_{i=1} c_i \left( 1 - \frac{r_i}{\bar{r}} \right)^2 }`

        where :math:`c_i` and :math:`r_i` are the fraction and radius of
        element :math:`i`, and :math:`\bar{r}` is the fraction-weighted
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

        # Compute the mean (1 - r/\bar{r})^2
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


class AtomicPackingEfficiency(BaseFeaturizer):
    """
    Packing efficiency based on a geometric theory of the amorphous packing
    of hard spheres.

    This featurizer computes two different kinds of the features. The first
    relate to the distance between a composition and the composition of
    the clusters of atoms expected to be efficiently packed based on a
    theory from
    `Laws et al.<http://www.nature.com/doifinder/10.1038/ncomms9123>`_.
    The second corresponds to the packing efficiency of a system if all atoms
    in the alloy are simultaneously as efficiently-packed as possible.

    The packing efficiency in these models is based on the Atomic Packing
    Efficiency (APE), which measures the difference between the ratio of
    the radii of the central atom to its neighbors and the ideal ratio
    of a cluster with the same number of atoms that has optimal packing
    efficiency. If the difference between the ratios is too large, the APE is
    positive. If the difference is too small, the APE is negative.

    Features:
        dist from {k} clusters |APE| < {thr} - The distance between an
            alloy composition and the k clusters that have a packing efficiency
            below thr from ideal
        mean simul. packing efficiency - Mean packing efficiency of all atoms.
            The packing efficiency is measured with respect to ideal (0)
        mean abs simul. packing efficiency - Mean absolute value of the
            packing efficiencies. Closer to zero is more efficiently packed

    References:
        [1] K.J. Laws, D.B. Miracle, M. Ferry, A predictive structural model
        for bulk metallic glasses, Nat. Commun. 6 (2015) 8123. doi:10.1038/ncomms9123.
    """

    def __init__(self, threshold=0.01, n_nearest=(1, 3, 5), max_types=6):
        """
        Initialize the featurizer

        Args:
            threshold (float):Threshold to use for determining whether
                a cluster is efficiently packed.
            n_nearest ({int}): Number of nearest clusters to use when considering features
            max_types (int): Maximum number of atom types to consider when
                looking for efficient clusters. The process for finding
                efficient clusters very expensive for large numbers of types
        """

        # Store the options
        self.threshold = threshold
        self.n_nearest = n_nearest
        self.max_types = max_types

        # Tool to convert composition objects to fractions as a vector
        self._el_frac = ElementFraction()

        # Get the number of elements in the output of `_el_frac`
        self._n_elems = len(self._el_frac.featurize(Composition('H')))

        # Tool for looking up radii
        self._data_source = MagpieData()

        # Lookup table of ideal radius ratios
        self.ideal_ratio = dict(
            [(3, 0.154701), (4, 0.224745), (5, 0.361654), (6, 0.414214),
             (7, 0.518145), (8, 0.616517), (9, 0.709914), (10, 0.798907),
             (11, 0.884003), (12, 0.902113), (13, 0.976006), (14, 1.04733),
             (15, 1.11632), (16, 1.18318), (17, 1.2481), (18, 1.31123),
             (19, 1.37271), (20, 1.43267), (21, 1.49119), (22, 1.5484),
             (23, 1.60436), (24, 1.65915)])

    def __hash__(self):
        return hash(self.threshold)

    def __eq__(self, other):
        if isinstance(other, AtomicPackingEfficiency):
            return self.get_params() == other.get_params()

    def featurize(self, comp):
        return list(self.compute_simultaneous_packing_efficiency(comp)) + \
               self.compute_nearest_cluster_distance(comp)

    def feature_labels(self):
        return ['mean simul. packing efficiency',
                'mean abs simul. packing efficiency'] + [
                   'dist from {} clusters |APE| < {:.3f}'.format(k,
                                                                 self.threshold)
                   for k in self.n_nearest]

    def citations(self):
        return ["@article{Laws2015,"
                "author = {Laws, K. J. and Miracle, D. B. and Ferry, M.},"
                "doi = {10.1038/ncomms9123},"
                "journal = {Nature Communications},"
                "pages = {8123},"
                "title = {{A predictive structural model for bulk metallic glasses}},"
                "url = {http://www.nature.com/doifinder/10.1038/ncomms9123},"
                "volume = {6},"
                "year = {2015}"]

    def implementors(self):
        return ['Logan Ward']

    def compute_simultaneous_packing_efficiency(self, comp):
        """Compute the packing efficiency of the system when the neighbor
        shell of each atom has the same composition as the alloy. When this
        criterion is satisfied, it is possible for every atom in this system
        to be simultaneously as efficiently-packed as possible.

        Args:
            comp (Composition): Composition to be assessed
        Returns
            (float) Average APE of all atoms
            (float) Average deviation of the APE of each atom from ideal (0)
        """

        # Compute the average atomic radius of the system
        elements, fractions = zip(*comp.element_composition.items())
        radii = self._data_source.get_elemental_properties(elements,
                                                           'MiracleRadius')
        mean_radius = PropertyStats.mean(radii, fractions)

        # Compute the APE for each cluster
        best_ape = [
            self.find_ideal_cluster_size(r / mean_radius)[1] for r in radii
        ]

        # Return the averages
        return PropertyStats.mean(best_ape, fractions), \
               PropertyStats.mean(np.abs(best_ape), fractions)

    def compute_nearest_cluster_distance(self, comp):
        """Compute the distance between a composition and that the nearest
        efficiently-packed clusters.

        Measures the mean :math:`L_2` distance between the alloy composition
        and the :math:`k`-nearest clusters with Atomic Packing Efficiencies
        within the user-specified tolerance of 1. :math:`k` is any of the
        numbers defined in the "n_nearest" parameter of this class.

        If there are less than `k` efficient clusters in the system, we use
        the maximum distance betweeen any two compositions (1) for the
        unmatched neighbors.

        Args:
            comp (Composition): Composition of material to evaluate
        Return:
            [float] Average distances
        """

        # Get the most common elements
        elems, _ = zip(*sorted(comp.element_composition.items(),
                                   key=lambda x: x[1], reverse=True))

        # Get the cluster lookup tool using the most common elements
        cluster_lookup = self.create_cluster_lookup_tool(
            elems[:self.max_types]
        )

        # Compute the composition vector
        comp_vec = self._el_frac.featurize(comp)

        # Compute the distances
        means = []
        for k in self.n_nearest:
            # Get the nearest clusters
            if cluster_lookup is None:
                dists = (np.array([]),)
                to_lookup = 0
            else:
                to_lookup = min(cluster_lookup._fit_X.shape[0], k)
                dists, _ = cluster_lookup.kneighbors([comp_vec], to_lookup)

            # Pad the list with 1's
            dists = dists[0].tolist() + [1]*(k - to_lookup)

            # Compute the average
            means.append(np.mean(dists))

        return means

    def create_cluster_lookup_tool(self, elements):
        """
        Get the compositions of efficiently-packed clusters in a certain system
        of elements

        Args:
            elements ([Element]): Elements in system
        Return:
            (NearNeighbors): Tool to find nearby clusters in this system. None
                if there are no efficiently-packed clusters for this combination of elements
        """
        elements = list(set(elements))
        return self._create_cluster_lookup_tool(tuple(sorted(elements)))

    @lru_cache()
    def _create_cluster_lookup_tool(self, elements):
        """
        Cached version of `create_cluster_lookup_tool`. Assumes that the
        elements are passed as sorted tuple with no duplicates

        Args:
            elements ([Element]): Elements in system
        Return:
            (NearNeighbors): Tool to find nearby clusters in this system. If
            there are no clusters, this class returns None
        """

        # Get the radii
        radii = self._data_source.get_elemental_properties(elements,
                                                           "MiracleRadius")

        # Get the maximum and minimum cluster sizes
        max_size = self.find_ideal_cluster_size(max(radii) / min(radii))[0]
        min_size = self.find_ideal_cluster_size(min(radii) / max(radii))[0]

        # Prepare a list to hold all possible clusters
        eff_clusters = []

        # Loop through all possible neighbor shells
        for size in range(min_size, max_size + 1):
            # Get the ideal radius ratio for a cluster of this size
            ideal_ratio = self.get_ideal_radius_ratio(size)

            # Get the mean radii and compositions of all possible
            #  combinations of elements in the neighbor shell
            s_radii = itertools.combinations_with_replacement(radii, size)
            s_elems = itertools.combinations_with_replacement(elements, size)

            #  Put the results in arrays for fast indexing
            mean_radii = np.array(list(s_radii)).mean(axis=1)
            s_elems = np.array(list(s_elems))

            # For each type of central atom, determine which have an APE
            #  within `self.threshold` of 1
            for center_radius, center_elem in zip(radii, elements):
                # Compute the APE of each cluster
                ape = 1 - np.divide(ideal_ratio, np.divide(center_radius,
                                                           mean_radii))

                # Get those which are within the threshold of 0
                #  and add their composition to the list of OK elements
                for hit in s_elems[np.abs(ape) < self.threshold]:
                    eff_clusters.append([center_elem] + hit.tolist())

        # Compute the composition vectors for all of the efficient clusters
        comps = np.zeros((len(eff_clusters), self._n_elems))
        for i, elems in enumerate(eff_clusters):
            for elem in elems:
                comps[i, elem.Z - 1] += 1
        comps = np.divide(comps, comps.sum(axis=1)[:, None])

        # Return tool to quickly determine distance from efficient clusters
        #  NearNeighbors requires at least 1 entry, so we return None if
        #   there are no nearby clusters
        return NearestNeighbors().fit(comps) if len(comps) > 0 else None

    def find_ideal_cluster_size(self, radius_ratio):
        """
        Get the optimal cluster size for a certain radius ratio

        Finds the number of nearest neighbors :math:`n` that minimizes
        :math:`|1 - rp(n)/r|`, where :math:`rp(n)` is the ideal radius
        ratio for a certain :math:`n` and :math:`r` is the actual ratio.

        Args:
            radius_ratio (float): math:`r / r_{neighbor}`
        Returns:
            (int) number of neighboring atoms for that will be the most
            efficiently packed.
            (float) Optimal APE
        """

        # Loop through cluster sizes from 3 to 24
        best_ape = np.inf
        best_n = None
        for n in range(3, 25):
            # Compute APE, check if it is the best
            ape = 1 - self.get_ideal_radius_ratio(n) / radius_ratio
            if abs(ape) < abs(best_ape):
                best_ape = ape
                best_n = n

            # If the APE is negative, this is either the best APE or
            #  We have already passed it
            if ape < 0:
                return best_n, best_ape

        return best_n, best_ape

    def get_ideal_radius_ratio(self, n_neighbors):
        """Compute the idea ratio between the central atom and neighboring
        atoms for a neighbor with a certain number of nearest neighbors.

        Based on work by `Miracle, Lord, and Ranganathan
        <https://www.jstage.jst.go.jp/article/matertrans/47/7/47_7_1737/_article/-char/en>`_.

        Args:
            n_neighbors (int): Number of atoms in 1st NN shell
        Return:
            (float) ideal radius ratio :math:`r / r_{neighbor}`
        """

        # NN must be in [3, 24]
        n = max(3, min(n_neighbors, 24))

        return self.ideal_ratio[n]
