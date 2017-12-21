from __future__ import division

from pymatgen import Element, MPRester
from pymatgen.core.periodic_table import get_el_sp

import os
import itertools
import collections

import numpy as np
import pandas as pd
import math
from functools import reduce

from matminer.featurizers.base import BaseFeaturizer
from matminer.utils.data import DemlData, MagpieData, PymatgenData, \
    CohesiveEnergyData
from matminer.featurizers.stats import PropertyStats

__author__ = 'Logan Ward, Jiming Chen, Ashwin Aggarwal, Kiran Mathew, ' \
             'Saurabh Bajaj, Qi Wang, Anubhav Jain'
module_dir = os.path.dirname(os.path.abspath(__file__))

class ElementProperty(BaseFeaturizer):
    """
    Class to calculate elemental property attributes. To initialize quickly,
    use the from_preset() method.

    Parameters:
        data_source (AbstractData or str): source from which to retrieve
            element property data (or use str for preset: "pymatgen",
            "magpie", or "deml")
        features (list of strings): List of elemental properties to use
            (these must be supported by data_source)
        stats (string): a list of weighted statistics to compute to for each
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

    @staticmethod
    def from_preset(preset_name):
        """
        Return ElementProperty from a preset string
        Args:
            preset_name: (str) can be one of "magpie", "deml", or "matminer"

        Returns:

        """
        if preset_name == "magpie":
            data_source = "magpie"
            features = ["Number", "MendeleevNumber", "AtomicWeight", "MeltingT",
                        "Column", "Row", "CovalentRadius",
                        "Electronegativity", "NsValence", "NpValence",
                        "NdValence", "NfValence", "NValance",
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
                        "first_ioniz", "total_ioniz", "electronegativity",
                        "formal_charge", "xtal_field_split",
                        "magn_moment", "so_coupling", "sat_magn",
                        "electric_pol", "GGAU_Etot", "mus_fere"]

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

        return ElementProperty(data_source, features, stats)

    def featurize(self, comp):
        """
        Get elemental property attributes

        Args:
            comp: Pymatgen composition object

        Returns:
            all_attributes: Specified property statistics of features
        """

        all_attributes = []

        for attr in self.features:
            elem_data = self.data_source.get_property(comp, attr)

            for stat in self.stats:
                all_attributes.append(
                    PropertyStats().calc_stat(elem_data, stat))

        return all_attributes

    def feature_labels(self):
        labels = []
        for attr in self.features:
            for stat in self.stats:
                labels.append("%s %s" % (stat, attr))

        return labels

    def citations(self):
        if self.data_source == "magpie":
            citation = (
                "@article{ward_agrawal_choudary_wolverton_2016, title={A general-purpose "
                "machine learning framework for predicting properties of inorganic materials}, "
                "volume={2}, DOI={10.1038/npjcompumats.2017.28}, number={1}, journal={npj "
                "Computational Materials}, author={Ward, Logan and Agrawal, Ankit and Choudhary, "
                "Alok and Wolverton, Christopher}, year={2016}}")
        elif self.data_source == "deml":
            citation = (
                "@article{deml_ohayre_wolverton_stevanovic_2016, title={Predicting density "
                "functional theory total energies and enthalpies of formation of metal-nonmetal "
                "compounds by linear regression}, volume={47}, DOI={10.1002/chin.201644254}, "
                "number={44}, journal={ChemInform}, author={Deml, Ann M. and Ohayre, Ryan and "
                "Wolverton, Chris and Stevanovic, Vladan}, year={2016}}")
        elif self.data_source == "pymatgen":
            citation = (
                "@article{Ong2013, author = {Ong, Shyue Ping and Richards, William Davidson and Jain, Anubhav and Hautier, "
                "Geoffroy and Kocher, Michael and Cholia, Shreyas and Gunter, Dan and Chevrier, Vincent L. and Persson, "
                "Kristin A. and Ceder, Gerbrand}, doi = {10.1016/j.commatsci.2012.10.028}, issn = {09270256}, "
                "journal = {Computational Materials Science}, month = {feb}, pages = {314--319}, "
                "publisher = {Elsevier B.V.}, title = {{Python Materials Genomics (pymatgen): A robust, open-source python "
                "library for materials analysis}}, url = {http://linkinghub.elsevier.com/retrieve/pii/S0927025612006295}, "
                "volume = {68}, year = {2013} } ")

        return citation

    def implementors(self):
        return ["Jiming Chen", "Logan Ward", "Anubhav Jain"]


class BandCenter(BaseFeaturizer):
    def featurize(self, comp):
        """
        (Rough) estimation of absolution position of band center using
        geometric mean of electronegativity.

        Args:
            comp: (Composition)

        Returns: (float) band center

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
    Calculate electronegativity difference between cations and anions
    (average, max, range, etc.)

    Parameters:
        data_source (data class): source from which to retrieve element data
        stats: Property statistics to compute

    Generates average electronegativity difference between cations and anions
    """

    def __init__(self, data_source=DemlData(), stats=None):
        self.data_source = data_source
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
        el_amt = comp.fractional_composition.get_el_amt_dict()

        best_guess = comp.oxi_state_guesses(max_sites=-1)[0]
        cations = [x for x in best_guess if best_guess[x] > 0]
        anions = [x for x in best_guess if best_guess[x] < 0]

        cation_en = [Element(x).X for x in cations]
        anion_en = [Element(x).X for x in anions]

        if len(cations) == 0 or len(anions) == 0:
            return len(self.stats) * [float("NaN")]

        avg_en_diff = []
        n_anions = sum([el_amt[el] for el in anions])

        # TODO: @WardLT, @JFChen3 I left this code as-is but am not quite sure what's going on. Why is there some normalization applied to anions but not cations? -computron
        for cat_en in cation_en:
            en_diff = 0
            for i in range(len(anions)):
                frac_anion = el_amt[anions[i]] / n_anions
                an_en = anion_en[i]
                en_diff += abs(cat_en - an_en) * frac_anion
            avg_en_diff.append(en_diff)

        cation_fracs = [el_amt[el] for el in cations]
        en_diff_stats = []

        for stat in self.stats:
            if stat == "std_dev":
                en_diff_stats.append(
                    PropertyStats().calc_stat(avg_en_diff, stat))
            else:
                en_diff_stats.append(
                    PropertyStats().calc_stat(avg_en_diff, stat,
                                              weights=cation_fracs))

        return en_diff_stats

    def feature_labels(self):

        labels = []
        for stat in self.stats:
            labels.append("%s EN difference" % stat)

        return labels

    def citations(self):
        citation = (
            "@article{deml_ohayre_wolverton_stevanovic_2016, title={Predicting density "
            "functional theory total energies and enthalpies of formation of metal-nonmetal "
            "compounds by linear regression}, volume={47}, DOI={10.1002/chin.201644254}, "
            "number={44}, journal={ChemInform}, author={Deml, Ann M. and Ohayre, Ryan and "
            "Wolverton, Chris and Stevanovic, Vladan}, year={2016}}")
        return citation

    def implementors(self):
        return ["Jiming Chen", "Logan Ward"]


class ElectronAffinity(BaseFeaturizer):
    """
    Class to calculate average electron affinity times formal charge of anion elements

    Parameters:
        data_source (data class): source from which to retrieve element data

    Generates average (electron affinity*formal charge) of anions
    """

    def __init__(self, data_source=DemlData()):
        self.data_source = data_source

    def featurize(self, comp):
        """
        Args:
            comp: Pymatgen Composition object

        Returns:
            avg_anion_affin (single-element list): average electron affinity*formal charge of anions
        """
        electron_affin = self.data_source.get_property(comp, "electron_affin",
                                                       combine_by_element=True)

        el_amt = comp.fractional_composition.get_el_amt_dict()
        elements = sorted(el_amt.keys(), key=lambda sym: get_el_sp(sym).Z)
        electron_affin = dict(zip(elements, electron_affin))

        oxi_states = comp.oxi_state_guesses(max_sites=-1)[0]

        avg_anion_affin = 0
        for i, el in enumerate(oxi_states):
            if oxi_states[el] < 0:
                avg_anion_affin += oxi_states[el] * electron_affin[el] * \
                                   el_amt[el]

        return [avg_anion_affin]

    def feature_labels(self):
        labels = ["avg anion electron affinity "]
        return labels

    def citations(self):
        citation = (
            "@article{deml_ohayre_wolverton_stevanovic_2016, title={Predicting density "
            "functional theory total energies and enthalpies of formation of metal-nonmetal "
            "compounds by linear regression}, volume={47}, DOI={10.1002/chin.201644254}, "
            "number={44}, journal={ChemInform}, author={Deml, Ann M. and Ohayre, Ryan and "
            "Wolverton, Chris and Stevanovic, Vladan}, year={2016}}")
        return citation

    def implementors(self):
        return ["Jiming Chen", "Logan Ward"]


class Stoichiometry(BaseFeaturizer):
    """
    Class to calculate stoichiometric attributes.

    Parameters:
        p_list (list of ints): list of norms to calculate
        num_atoms (bool): whether to return number of atoms
    """

    def __init__(self, p_list=[0, 2, 3, 5, 7, 10], num_atoms=False):
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

        n_atoms = comp.num_atoms

        if self.p_list == None:
            stoich_attr = [n_atoms]  # return num atoms if no norms specified
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
                stoich_attr = [n_atoms] + p_norms
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
        citation = (
            "@article{ward_agrawal_choudary_wolverton_2016, title={A general-purpose "
            "machine learning framework for predicting properties of inorganic materials}, "
            "volume={2}, DOI={10.1038/npjcompumats.2017.28}, number={1}, journal={npj "
            "Computational Materials}, author={Ward, Logan and Agrawal, Ankit and Choudhary, "
            "Alok and Wolverton, Christopher}, year={2016}}")
        return citation

    def implementors(self):
        return ["Jiming Chen", "Logan Ward"]


class ValenceOrbital(BaseFeaturizer):
    """
        Class to calculate valence orbital attributes

        Parameters:
            data_source (data object): source from which to retrieve element data
            orbitals (list): orbitals to calculate
            props (list): specifies whether to return average number of electrons in each orbital,
                fraction of electrons in each orbital, or both
    """

    def __init__(self, data_source=MagpieData(), orbitals=["s", "p", "d", "f"],
                 props=["avg", "frac"]):
        self.data_source = data_source
        self.orbitals = orbitals
        self.props = props

    def featurize(self, comp):
        """Weighted fraction of valence electrons in each orbital

           Args:
                comp: Pymatgen composition object

           Returns:
                valence_attributes (list of floats): Average number and/or fraction of valence electrons in specfied orbitals
        """

        el_amt = comp.fractional_composition.get_el_amt_dict()
        elements = sorted(el_amt.keys(), key=lambda sym: get_el_sp(sym).Z)
        el_fracs = [el_amt[el] for el in elements]

        avg = []

        for orb in self.orbitals:
            avg.append(
                PropertyStats().mean(
                    self.data_source.get_property(comp, "N%sValence" % orb,
                                                  combine_by_element=True),
                    weights=el_fracs))

        if "frac" in self.props:
            avg_total_valence = PropertyStats().mean(
                self.data_source.get_property(comp, "NValance",
                                              combine_by_element=True),
                weights=el_fracs)
            frac = [a / avg_total_valence for a in avg]

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
    Class to calculate ionic property attributes

    Parameters:
        data_source (data class): source from which to retrieve element data
    """

    def __init__(self, data_source=MagpieData()):
        self.data_source = data_source

    def featurize(self, comp):
        """
        Ionic character attributes

        Args:
            comp: Pymatgen composition object

        Returns:
            cpd_possible (bool): Indicates if a neutral ionic compound is possible
            max_ionic_char (float): Maximum ionic character between two atoms
            avg_ionic_char (float): Average ionic character
        """

        el_amt = comp.get_el_amt_dict()
        elements = sorted(el_amt.keys(), key=lambda sym: get_el_sp(sym).Z)
        values = [el_amt[el] for el in elements]

        if len(elements) < 2:  # Single element
            cpd_possible = True
            max_ionic_char = 0
            avg_ionic_char = 0
        else:
            # Get magpie data for each element
            ox_states = self.data_source.get_property(comp, "OxidationStates",
                                                      combine_by_element=True)
            elec = self.data_source.get_property(comp, "Electronegativity",
                                                 combine_by_element=True)

            # TODO: consider replacing with oxi_state_guesses - depends on
            # whether Magpie oxidation states table matches oxi_states table

            # Determine if neutral compound is possible
            cpd_possible = False
            ox_sets = itertools.product(*ox_states)
            for ox in ox_sets:
                if abs(np.dot(ox, values)) < 1e-4:
                    cpd_possible = True
                    break

            # Ionic character attributes
            atom_pairs = itertools.combinations(range(len(elements)), 2)
            el_frac = list(np.true_divide(values, sum(values)))

            ionic_char = []
            avg_ionic_char = 0

            for pair in atom_pairs:
                XA = elec[pair[0]]
                XB = elec[pair[1]]
                ionic_char.append(1.0 - np.exp(-0.25 * ((XA - XB) ** 2)))
                avg_ionic_char += el_frac[pair[0]] * el_frac[pair[1]] * \
                                  ionic_char[-1]

            max_ionic_char = np.max(ionic_char)

        return list((cpd_possible, max_ionic_char, avg_ionic_char))

    def feature_labels(self):
        labels = ["compound possible", "max ionic char", "avg ionic char"]
        return labels

    def citations(self):
        citation = (
            "@article{ward_agrawal_choudary_wolverton_2016, title={A general-purpose "
            "machine learning framework for predicting properties of inorganic materials}, "
            "volume={2}, DOI={10.1038/npjcompumats.2017.28}, number={1}, journal={npj "
            "Computational Materials}, author={Ward, Logan and Agrawal, Ankit and Choudhary, "
            "Alok and Wolverton, Christopher}, year={2016}}")
        return citation

    def implementors(self):
        return ["Jiming Chen", "Logan Ward"]


# TODO: is this descriptor useful or just noise?
class ElementFraction(BaseFeaturizer):
    """
    Class to calculate the atomic fraction of each element in a composition.

    Generates: vector where each index represents an element in atomic number order.
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
        return ["Ashwin Aggarwal, Logan Ward"]


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
        citation = (
            "@article{deml_ohayre_wolverton_stevanovic_2016, title={Predicting density "
            "functional theory total energies and enthalpies of formation of metal-nonmetal "
            "compounds by linear regression}, volume={47}, DOI={10.1002/chin.201644254}, "
            "number={44}, journal={ChemInform}, author={Deml, Ann M. and Ohayre, Ryan and "
            "Wolverton, Chris and Stevanovic, Vladan}, year={2016}}")
        return citation

    def implementors(self):
        return ["Jiming Chen, Logan Ward"]


# TODO: why is this a "feature" of a compound? Seems more like a pymatgen analysis thing?
#   Deml *et al.* PRB 085142 use the FERE correction energy as the basis for features. I've adjusted the class
#    documentation to make it clearer that this class computes features related to the FERE correction values, and
#    not that it is performing some kind of thermodynamic analysis. Really, we should pre-compute these correction
#    values and combine this with the ElementProperty class.
class FERECorrection(BaseFeaturizer):
    """
    Class to calculate features related to the difference between fitted elemental-phase reference
    energy (FERE) and GGA+U energy

    Parameters:
        data_source (data class): source from which to retrieve element data
        stats: Property statistics to compute

    Generates: Property statistics of difference between FERE and GGA+U energy
    """

    def __init__(self, data_source=DemlData(), stats=None):
        self.data_source = data_source
        if stats == None:
            self.stats = ["minimum", "maximum", "range", "mean", "std_dev"]
        else:
            self.stats = stats

    def featurize(self, comp):
        """
        Args:
            comp: Pymatgen Composition object

        Returns:
            fere_corr_stats (list of floats): Property stats of FERE correction
        """

        el_amt = comp.fractional_composition.get_el_amt_dict()
        elements = sorted(el_amt.keys(), key=lambda sym: get_el_sp(sym).Z)
        el_frac = [el_amt[el] for el in elements]

        GGAU_Etot = self.data_source.get_property(comp, "GGAU_Etot",
                                                  combine_by_element=True)
        mus_fere = self.data_source.get_property(comp, "mus_fere",
                                                 combine_by_element=True)

        fere_corr = [mus_fere[i] - GGAU_Etot[i] for i in range(len(GGAU_Etot))]

        fere_corr_stats = []
        for stat in self.stats:
            fere_corr_stats.append(
                PropertyStats().calc_stat(fere_corr, stat, weights=el_frac))

        return fere_corr_stats

    def feature_labels(self):

        labels = []
        for stat in self.stats:
            labels.append("%s FERE correction" % stat)

        return labels

    def citations(self):
        citation = (
            "@article{deml_ohayre_wolverton_stevanovic_2016, title={Predicting density "
            "functional theory total energies and enthalpies of formation of metal-nonmetal "
            "compounds by linear regression}, volume={47}, DOI={10.1002/chin.201644254}, "
            "number={44}, journal={ChemInform}, author={Deml, Ann M. and Ohayre, Ryan and "
            "Wolverton, Chris and Stevanovic, Vladan}, year={2016}}")
        return citation

    def implementors(self):
        return ["Jiming Chen", "Logan Ward"]


class CohesiveEnergy(BaseFeaturizer):

    def __init__(self, mapi_key=None):
        """
        Class to get cohesive energy per atom of a compound by adding known
        elemental cohesive energies from the formation energy of the
        compound.

        Parameters:
            mapi_key (str): Materials API key for looking up formation energy
                by composition alone (if you don't set the formation energy
                yourself).
        """
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
                most_stable_entry = sorted(struct_lst,
                                           key=lambda e:
                                           e['energy_per_atom'])[0]
                formation_energy_per_atom = most_stable_entry[
                    'formation_energy_per_atom']
            else:
                raise ValueError('No structure found in MP for {}'.format(comp))

        # Subtract elemental cohesive energies from formation energy
        cohesive_energy = -formation_energy_per_atom * comp.num_atoms
        for el in el_amt_dict:
            cohesive_energy += el_amt_dict[el] * \
                               CohesiveEnergyData().get_property(el)

        cohesive_energy_per_atom = cohesive_energy / comp.num_atoms

        return [cohesive_energy_per_atom]

    def feature_labels(self):
        return ["cohesive energy"]

    def implementors(self):
        return ["Saurabh Bajaj", "Anubhav Jain"]

    def citations(self):
        # TODO: @sbajaj unclear whether cohesive energies are taken from first ref, second ref, or combination of both
        return [
            "@misc{, title = {{Knowledgedoor Cohesive energy handbook}}, "
            "url = {http://www.knowledgedoor.com/2/elements{\_}handbook/cohesive{\_}energy.html}}",
            "@book{Kittel, author = {Kittel, C}, isbn = {978-0-471-41526-8}, "
            "publisher = {Wiley}, title = {{Introduction to Solid State "
            "Physics, 8th Edition}}, year = {2005}}"]


# TODO: read data file only once!! (on init, then store)
# TODO: general code review, typo fixes, etc
class Miedema(BaseFeaturizer):
    """
    Class to calculate the formation enthalpies of the intermetallic compound,
    solid solution and amorphous phase of a given composition, based on the
    semi-empirical Miedema model (and some extensions), particularly for
    transitional metal alloys.

    Support elemental, binary and multicomponent alloys.
        For elemental/binary alloys, the formulation is based on the original
        works by Miedema et al. in 1980s;
        For multicomponent alloys, the formulation is basically linear combination
        of sub-binary systems. This is reported to work well for ternary alloys,
        but needs to be careful with quaternary alloys and more.

    Args:
        struct_types (str / list of str): default='inter'
            if str, one target structure;
            if list, a list of target structures.

            'inter': intermetallic compound
            'ss': solid solution
            'amor': amorphous phase
            'all': same for ['inter', 'ss', 'amor']
            ['inter', 'ss']: amorphous phase and solid solution, as an example

        ss_types (str / list of str): default='min', only for ss
            if str, one structure type of ss;
            if list, a list of structure types of ss.

            'fcc': fcc solid solution
            'bcc': bcc solid solution
            'hcp': hcp solid solution
            'no_latt': solid solution with no specific structure type
            'min': min value of ['fcc', 'bcc', 'hcp', 'no_latt']
            'all': same for ['fcc', 'bcc', 'hcp', 'no_latt']
            ['fcc', 'bcc']: fcc and bcc solid solutions, as an example

        data_source (str): default='Miedema', source of dataset
            -'Miedema': read from 'Miedema.csv'
                        parameterized by Miedema et al. in 1980s,
                        containing following parameters for 73 types of elements:
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
        a list of Miedema formation enthalpies (per atom) of target structures
        for a given composition:
            formation_enthalpy_inter: for interatomic compound
            formation_enthalpy_ss: for solid solution, can be divided into
                                   'min', 'fcc', 'bcc', 'hcp', 'no_latt'
                                    for different lattice_types
            formation_enthalpy_amor: for amorphous phase

    """

    data_dir = os.path.join(module_dir, "..", "utils", "data_files")

    def __init__(self, struct_types='inter', ss_types='min', data_source='Miedema'):
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
                os.path.join(self.data_dir, 'Miedema.csv'), index_col='element')
        else:
            raise NotImplementedError('data_source {} not implemented yet'.
                                      format(self, data_source))

    def deltaH_chem(self, elements, fracs, struct):
        """chemical term of formation enthalpy

        Args:
            elements (list of str): list of elements
            fracs (list of floats): list of atomic fractions
            struct (str): 'inter', 'ss' or 'amor'

        Returns:
            deltaH_chem (float): chemical term of formation enthalpy

        """
        for el in elements:
            if el not in self.df_dataset.index:
                return np.nan
        df_el = self.df_dataset.loc[elements]
        v_molar = np.array(df_el['molar_volume'])
        n_ws = np.array(df_el['electron_density'])
        elec = np.array(df_el['electronegativity'])
        val = np.array(df_el['valence_electrons'])
        a = np.array(df_el['a_const'])
        R = np.array(df_el['R_const'])
        H_trans = np.array(df_el['H_trans'])

        if struct == 'inter':
            gamma = 8
        elif struct == 'amor':
            gamma = 5
        else:
            gamma = 0

        c_surf = (fracs * np.power(v_molar, 2/3) /
                  np.dot(fracs, np.power(v_molar, 2/3)))

        f = (c_surf * (1 +
                       gamma * np.power(np.multiply.reduce(c_surf), 2)))[::-1]

        v_alloy = np.array([np.power(v_molar[0], 2/3) *
                            (1 + a[0] * f[0] * (elec[0] - elec[1])),
                            np.power(v_molar[1], 2/3) *
                            (1 + a[1] * f[1] * (elec[1] - elec[0]))])

        c_surf_alloy = fracs * v_alloy / np.dot(fracs, v_alloy)

        f_alloy = (c_surf_alloy *
                   (1 + gamma *
                    np.power(np.multiply.reduce(c_surf_alloy), 2)))[::-1]

        threshold = range(3, 12)
        if (val[0] in threshold) and (val[1] in threshold):
            P = 14.10
            R = 0.00

        elif (val[0] not in threshold) and (val[1] not in threshold):
            P = 10.70
            R = 0.00

        else:
            P = 12.35
            R = np.multiply.reduce(R) * P
        Q = P * 9.40

        eta_ab = (2 * (-P * np.power(elec[0] - elec[1], 2) - R +
                       Q * np.power(np.power(n_ws[0], 1/3) -
                                    np.power(n_ws[1], 1/3), 2)) /
                  reduce(lambda x, y: 1/x + 1/y, np.power(n_ws, 1/3)))

        deltaH_chem = (f_alloy[0] * fracs[0] * v_alloy[0] * eta_ab +
                       np.dot(fracs, H_trans))

        return deltaH_chem

    def deltaH_elast(self, elements, fracs):
        """elastic term of formation enthalpy

        Args:
            elements (list of str): list of elements
            fracs (list of floats): list of atomic fractions

        Returns:
            deltaH_elastic (float): elastic term of formation enthalpy

        """
        for el in elements:
            if el not in self.df_dataset.index:
                return np.nan
        df_el = self.df_dataset.loc[elements]
        v_molar = np.array(df_el['molar_volume'])
        n_ws = np.array(df_el['electron_density'])
        elec = np.array(df_el['electronegativity'])
        compr = np.array(df_el['compressibility'])
        shear_mod = np.array(df_el['shear_modulus'])

        alp = (1.5 * np.power(v_molar, 2/3) /
               reduce(lambda x, y: 1/x + 1/y, np.power(n_ws, 1/3)))

        v_alloy = (v_molar +
                   np.array([alp[0] * (elec[0] - elec[1]) / n_ws[0],
                             alp[1] * (elec[1] - elec[0]) / n_ws[1]]))

        alp_alloy = (1.5 * np.power(v_alloy, 2/3) /
                     reduce(lambda x, y: 1/x + 1/y, np.power(n_ws, 1/3)))

        # effective volume in alloy
        Vab_alloy = (v_molar[0] +
                     np.array([alp_alloy[0] * (elec[0] - elec[1]) / n_ws[0],
                               alp_alloy[1] * (elec[1] - elec[0]) / n_ws[0]]))

        Vba_alloy = (v_molar[1] +
                     np.array([alp_alloy[0] * (elec[0] - elec[1]) / n_ws[1],
                               alp_alloy[1] * (elec[1] - elec[0]) / n_ws[1]]))

        # H_elast A in B
        Hab_elast = ((2 * compr[0] * shear_mod[1] *
                      np.power((Vab_alloy[0] - Vba_alloy[0]), 2)) /
                     (4 * shear_mod[1] * Vab_alloy[0] +
                      3 * compr[0] * Vba_alloy[0]))

        # H_elast B in A
        Hba_elast = ((2 * compr[1] * shear_mod[0] *
                      np.power((Vba_alloy[1] - Vab_alloy[1]), 2)) /
                     (4 * shear_mod[0] * Vba_alloy[1] +
                      3 * compr[1] * Vab_alloy[1]))

        deltaH_elast = (np.multiply.reduce(fracs) *
                        (fracs[1] * Hab_elast + fracs[0] * Hba_elast))

        return deltaH_elast

    def deltaH_struct(self, elements, fracs, latt):
        """structural term of formation enthalpy, only for solid solution

        Args:
            elements (list of str): list of elements
            fracs (list of floats): list of atomic fractions
            latt (str): 'fcc', 'bcc', 'hcp' or 'no_latt'

        Returns:
            deltaH_struct (float): structural term of formation enthalpy

        """
        for el in elements:
            if el not in self.df_dataset.index:
                return np.nan
        df_el = self.df_dataset.loc[elements]
        val = np.array(df_el['valence_electrons'])
        struct_stab = np.array(df_el['structural_stability'])

        if latt == 'fcc':
            latt_stab_dict = {0.0: 0., 1.0: 0, 2.0: 0, 3.0: -2, 4.0: -1.5,
                              5.0: 9., 5.5: 14., 6.0: 11., 7.0: -3., 8.0: -9.5,
                              8.5: -11., 9.0: -9., 10.0: -2., 11.0: 1.5,
                              12.0: 0., 13.0: 0., 14.0: 0., 15.0: 0.}
        elif latt == 'bcc':
            latt_stab_dict = {0.0: 0., 1.0: 0., 2.0: 0., 3.0: 2.2, 4.0: 2.,
                              5.0: -9.5, 5.5: -14.5, 6.0: -12., 7.0: 4.,
                              8.0: 10., 8.5: 11., 9.0: 8.5, 10.0: 1.5,
                              11.0: 1.5, 12.0: 0., 13.0: 0., 14.0: 0., 15.0: 0.}
        elif latt == 'hcp':
            latt_stab_dict = {0.0: 0., 1.0: 0., 2.0: 0., 3.0: -2.5, 4.0: -2.5,
                              5.0: 10., 5.5: 15., 6.0: 13., 7.0: -5.,
                              8.0: -10.5, 8.5: -11., 9.0: -8., 10.0: -1.,
                              11.0: 2.5, 12.0: 0., 13.0: 0., 14.0: 0., 15.0: 0.}
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
        """topological term of formation enthalpy, only for amorphous phase

        Args:
            elements (list of str): list of elements
            fracs (list of floats): list of atomic fractions

        Returns:
            deltaH_topo (float): topological term of formation enthalpy

        """
        for el in elements:
            if el not in self.df_dataset.index:
                return np.nan
        else:
            df_el = self.df_dataset.loc[elements]
            melt_point = np.array(df_el['melting_point'])

        deltaH_topo = 3.5 * np.dot(fracs, melt_point) / 1000
        return deltaH_topo

    def featurize(self, comp):
        """Get Miedema formation enthalpies of target structures: inter, amor,
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
        len_elements = len(elements)
        # divide into a list of sub-binaries
        element_bins = []
        frac_bins = []
        for i in range(len_elements - 1):
            for j in range(i+1, len_elements):
                element_bins.append([elements[i], elements[j]])
                frac_bins.append([fracs[i], fracs[j]])

        miedema = []
        for struct_type in self.struct_types:
            # inter: intermetallic compound
            if struct_type == 'inter':
                deltaH_chem_inter = 0
                for i_inter, element_bin in enumerate(element_bins):
                    deltaH_chem_inter += self.deltaH_chem(element_bin,
                                                          frac_bins[i_inter],
                                                          'inter')
                miedema.append(deltaH_chem_inter)
            # ss: solid solution
            elif struct_type == 'ss':
                deltaH_chem_ss = 0
                deltaH_elast_ss = 0
                for bin, element_bin in enumerate(element_bins):
                    deltaH_chem_ss += self.deltaH_chem(element_bin,
                                                       frac_bins[bin], 'ss')
                    deltaH_elast_ss += self.deltaH_elast(element_bin,
                                                         frac_bins[bin])

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
                        deltaH_struct_ss = self.deltaH_struct(elements, fracs, ss_type)
                        miedema.append(deltaH_chem_ss + deltaH_elast_ss + deltaH_struct_ss)

            # amor: amorphous phase
            elif struct_type == 'amor':
                deltaH_chem_amor = 0
                deltaH_topo_amor = self.deltaH_topo(elements, fracs)
                for bin, element_bin in enumerate(element_bins):
                    deltaH_chem_amor += self.deltaH_chem(element_bin,
                                                         frac_bins[bin], 'amor')
                miedema.append(deltaH_chem_amor + deltaH_topo_amor)

        # convert kJ/mol to eV/atom. The original Miedema model is in kJ/mol.
        miedema = [deltaH / 96.4853 for deltaH in miedema]
        return miedema

    def feature_labels(self):
        labels = []
        for struct_type in self.struct_types:
            if struct_type == 'ss':
                for ss_type in self.ss_types:
                    labels.append('formation_enthalpy_ss_'+ss_type)
            else:
                labels.append('formation_enthalpy_'+struct_type)
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
