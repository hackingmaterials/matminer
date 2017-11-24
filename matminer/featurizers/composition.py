from __future__ import division

from pymatgen import Element, Composition, MPRester
from pymatgen.core.periodic_table import get_el_sp

import os
import json
import itertools

import numpy as np
import pandas as pd
import math
from functools import reduce

from matminer.featurizers.base import BaseFeaturizer
from matminer.featurizers.data import DemlData, MagpieData, PymatgenData, \
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


class Miedema(BaseFeaturizer):
    """
    Class to calculate the formation enthalpies of the intermetallic compound,
    solid solution and amorphous phase of a given composition, based on the
    semi-empirical Miedema model for transitional metals.
    (use the original formulation in 1980s, see citation)

    **Currently only elementary or binary composition is supported, may extend to ternary or more later.

    Parameters:
        struct (String): one target structure or a list of target structures separated by '|'
                          'inter'    :   intermetallic compound --by default
                          'ss'       :   solid solution
                          'amor'     :   amorphous phase
                          'inter|ss' :   intermetallic compound and solid solution, as an example
                          'all'      :   same for 'inter|ss|amor'

                           for 'ss', one can designate the lattice type: if entering 'ss, bcc', 'ss, fcc', 'ss, hcp',
                           then the lattice type of ss is fixed; if not, returning the minimum formation enthalpy of
                           possible lattice types


        dataset (String): source of parameters:
                           'Miedema': the original paramerization by Miedema et al. in 1989
                           'MP': extract some features from MP to replace the original ones in 'Miedema'
                           'Citrine': extract some features from Citrine to replace the original ones in 'Miedema'
                           **Currently not done yet
    """

    def __init__(self, struct='inter', dataset='Miedema'):
        if struct == 'all':
            struct = 'inter|amor|ss'
        self.struct = struct
        self.dataset = dataset

    # chemical term of formation enthalpy
    def delta_H_chem(self, elements, fracs, struct):
        if self.dataset == 'Miedema':
            df_dataset = pd.read_csv(os.path.join(module_dir, 'data_files', 'Miedema.csv'), index_col='element')
            for element in elements:
                if element not in df_dataset.index:
                    return np.nan
            df_element = df_dataset.loc[elements]
            V_molar = np.array(df_element['molar_volume'])
            n_WS = np.array(df_element['electron_density'])
            elec = np.array(df_element['electronegativity'])
            valence = np.array(df_element['valence_electrons'])

            a_const = np.array(df_element['a_const'])
            R_const = np.array(df_element['R_const'])
            H_trans = np.array(df_element['H_trans'])
        else:
            # allow to extract parameters for ab initio databases eg MP, Citrine ** Currently not done
            df_dataset = pd.read_csv(os.path.join(module_dir, 'data_files', 'Miedema.csv'), index_col='element')
            df_element = df_dataset.ix[elements]
            V_molar = np.array(df_element['molar_volume'])
            n_WS = np.array(df_element['electron_density'])
            elec = np.array(df_element['electronegativity'])
            valence = np.array(df_element['valence_electrons'])

            a_const = np.array(df_element['a_const'])
            R_const = np.array(df_element['R_const'])
            H_trans = np.array(df_element['H_trans'])

        if struct == 'inter':
            gamma = 8
        elif struct == 'amor':
            gamma = 5
        else:
            gamma = 0

        c_surf = fracs * np.power(V_molar, 2/3) / np.dot(fracs, np.power(V_molar, 2/3))
        f = (c_surf * (1 + gamma * np.power(np.multiply.reduce(c_surf), 2)))[::-1]
        V_alloy = np.array([np.power(V_molar[0], 2/3) * (1 + a_const[0] * f[0] * (elec[0] - elec[1])),
                            np.power(V_molar[1], 2/3) * (1 + a_const[1] * f[1] * (elec[1] - elec[0]))])

        c_surf_alloy = fracs * V_alloy / np.dot(fracs, V_alloy)

        f_alloy = (c_surf_alloy * (1 + gamma * np.power(np.multiply.reduce(c_surf_alloy), 2)))[::-1]

        threshold = range(3,12)

        if (valence[0] in threshold and valence[1] in threshold):
            P_const = 14.10
            R_const = 0.00

        elif (valence[0] not in threshold) and (valence[1] not in threshold):
            P_const = 10.70
            R_const = 0.00

        else:
            P_const = 12.35
            R_const = np.multiply.reduce(R_const) * P_const

        Q_const = P_const * 9.40

        eta_AB = 2 * (-P_const * np.power(elec[0] - elec[1], 2) + Q_const * np.power(np.power(n_WS[0], 1/3) -
                 np.power(n_WS[1], 1/3), 2) - R_const) / reduce(lambda x,y: 1/x + 1/y,np.power(n_WS, 1/3))

        delta_H_chem = f_alloy[0] * fracs[0] * V_alloy[0] * eta_AB + np.dot(fracs, H_trans)
        return delta_H_chem

    # elastic term of formation enthalpy
    def delta_H_elast(self, elements, fracs):
        if self.dataset == 'Miedema':
            df_dataset = pd.read_csv(os.path.join(module_dir, 'data_files', 'Miedema.csv'),index_col='element')
            for element in elements:
                if element not in df_dataset.index:
                    return np.nan
            df_element = df_dataset.loc[elements]
            V_molar = np.array(df_element['molar_volume'])
            n_WS = np.array(df_element['electron_density'])
            elec = np.array(df_element['electronegativity'])
            compr = np.array(df_element['compressibility'])
            shear_mod = np.array(df_element['shear_modulus'])
        else:
            # allow to extract parameters for ab initio databases eg MP, Citrine ** Currently not done
            df_dataset = pd.read_csv(os.path.join(module_dir, 'data_files', 'Miedema.csv'),index_col='element')
            df_element = df_dataset.ix[elements]
            V_molar = np.array(df_element['molar_volume'])
            n_WS = np.array(df_element['electron_density'])
            elec = np.array(df_element['electronegativity'])
            compr = np.array(df_element['compressibility'])
            shear_mod = np.array(df_element['shear_modulus'])

        alpha_pure = 1.5 * np.power(V_molar, 2/3) / reduce(lambda x,y: 1/x + 1/y, np.power(n_WS, 1/3))

        # volume correction
        V_alloy = V_molar + np.array([alpha_pure[0] * (elec[0] - elec[1]) / n_WS[0],
                                      alpha_pure[1] * (elec[1] - elec[0]) / n_WS[1]])

        alpha_alloy = 1.5 * np.power(V_alloy, 2/3) / reduce(lambda x, y: 1/x + 1/y, np.power(n_WS, 1/3))

        # effective volume in alloy
        V_alloy_AB = V_molar[0] + np.array([alpha_alloy[0] * (elec[0] - elec[1]) / n_WS[0],
                                            alpha_alloy[1] * (elec[1] - elec[0]) / n_WS[0]])
        V_alloy_BA = V_molar[1] + np.array([alpha_alloy[0] * (elec[0] - elec[1]) / n_WS[1],
                                            alpha_alloy[1] * (elec[1] - elec[0]) / n_WS[1]])

        # H_elast A in B
        H_elast_AB = 2 * compr[0] * shear_mod[1] * np.power((V_alloy_AB[0] - V_alloy_BA[0]), 2) \
                     / (4 * shear_mod[1] * V_alloy_AB[0] + 3 * compr[0] * V_alloy_BA[0])
        # H_elast B in A
        H_elast_BA = 2 * compr[1] * shear_mod[0] * np.power((V_alloy_BA[1] - V_alloy_AB[1]), 2) \
                     / (4 * shear_mod[0] * V_alloy_BA[1] + 3 * compr[1] * V_alloy_AB[1])

        delta_H_elast = np.multiply.reduce(fracs) * (fracs[1] * H_elast_AB + fracs[0] * H_elast_BA)

        return delta_H_elast

    # structural term of formation enthalpy
    def delta_H_struct(self, elements, fracs, lattice):
        if self.dataset == 'Miedema':
            df_dataset = pd.read_csv(os.path.join(module_dir, 'data_files', 'Miedema.csv'),index_col='element')
            for element in elements:
                if element not in df_dataset.index:
                    return np.nan
            df_element = df_dataset.loc[elements]
            valence = np.array(df_element['valence_electrons'])
            struct_stability = np.array(df_element['structural_stability'])

        else:
            # allow to extract parameters for ab initio databases eg MP, Citrine **Currently not done
            df_dataset = pd.read_csv(os.path.join(module_dir, 'data_files', 'Miedema.csv'),index_col='element')
            df_element = df_dataset.ix[elements]
            valence = np.array(df_element['valence_electrons'])
            struct_stability = np.array(df_element['structural_stability'])

        # fcc
        if lattice == 'fcc':
            latt_stability_dict = {0.0: 0, 1.0: 0, 2.0: 0, 3.0: -2, 4.0: -1.5, 5.0: 9, 5.5: 14, 6.0: 11, 7.0: -3,
                                   8.0: -9.5, 8.5: -11,9.0: -9, 10.0: -2, 11.0: 1.5, 12.0: 0, 13.0: 0, 14.0: 0, 15.0: 0}
        # bcc
        elif lattice == 'bcc':
            latt_stability_dict = {0.0: 0, 1.0: 0, 2.0: 0, 3.0: 2.2, 4.0: 2, 5.0: -9.5, 5.5: -14.5, 6.0: -12, 7.0: 4,
                                   8.0: 10, 8.5: 11, 9.0: 8.5, 10.0: 1.5, 11.0: 1.5, 12.0: 0, 13.0: 0, 14.0: 0, 15.0: 0}
        # hcp
        elif lattice == 'hcp':
            latt_stability_dict = {0.0: 0, 1.0: 0, 2.0: 0, 3.0: -2.5, 4.0: -2.5, 5.0: 10, 5.5: 15, 6.0: 13, 7.0: -5,
                                   8.0: -10.5, 8.5: -11, 9.0: -8, 10.0: -1, 11.0: 2.5, 12.0: 0, 13.0: 0, 14.0: 0, 15.0: 0}
        else:
            return 0

        # lattice stability of different structures: fcc, bcc, hcp
        valence_avg = np.dot(fracs, valence)
        valence_boundary_lower, valence_boundary_upper = 0, 0
        for key in latt_stability_dict.keys():
            if valence_avg - key <= 0:
                valence_boundary_upper = key
                break
            else:
                valence_boundary_lower = key

        latt_stability = (valence_avg - valence_boundary_lower) * latt_stability_dict[valence_boundary_upper] / \
                         (valence_boundary_upper - valence_boundary_lower) + (valence_boundary_upper - valence_avg) * \
                         latt_stability_dict[valence_boundary_lower] / (valence_boundary_upper - valence_boundary_lower)

        delta_H_struct = latt_stability - np.dot(fracs, struct_stability)
        return delta_H_struct

    # entropy term of Gibbs free energy
    # currently not used, only return enthalpy term
    def delta_S(self, fracs):
        frac_sum = 0
        for frac in fracs:
            if frac == 0:
                frac_sum += 0
            else:
                frac_sum += frac * math.log(frac)
        delta_S = -8.314 * frac_sum / 1000
        return delta_S

    def featurize(self, comp):
        """
        Get Miedema formation enthalpy of target structures
        :param comp: Pymatgen composition object
        :return: delta_H_inter :  formation enthalpy of intermetallic compound
                 delta_H_ss    :  formation enthalpy of solid solution
                 delta_H_amor  :  formation enthalpy of amorphous phase
        """

        el_amt = comp.fractional_composition.get_el_amt_dict()
        elements = sorted(el_amt.keys(), key=lambda sym: get_el_sp(sym).X)
        fracs = [el_amt[el] for el in elements]

        if self.dataset == 'Miedema':
            df_dataset = pd.read_csv(os.path.join(module_dir, 'data_files', 'Miedema.csv'), index_col='element')
            for element in elements:
                if element not in df_dataset.index:
                    melting_point = [np.nan,np.nan]
                    break
            else:
                df_element = df_dataset.loc[elements]
                melting_point = np.array(df_element['melting_point'])

        else:
            # allow to extract parameters for ab initio databases eg MP, Citrine **Currently not done
            df_dataset = pd.read_csv(os.path.join(module_dir, 'data_files', 'Miedema.csv'), index_col='element')
            df_element = df_dataset.ix[elements]
            melting_point = np.array(df_element['melting_point'])

        miedema_result = []
        for struct_now in self.struct.split('|'):
            # inter: intermetallic compound
            if struct_now == 'inter':
                delta_H_chem_inter = self.delta_H_chem(elements, fracs, 'inter')
                delta_H_inter = delta_H_chem_inter
                miedema_result.append(delta_H_inter)

            # ss: solid solution, four types of solid solution prototypes, default: minimum
            elif struct_now.startswith('ss'):
                delta_H_chem_ss = self.delta_H_chem(elements, fracs, 'ss')
                delta_H_elast = self.delta_H_elast(elements, fracs)

                if struct_now != 'ss':
                    lattice = struct_now.split(',')
                    for i in range(1, len(lattice)):
                        miedema_result.append(delta_H_chem_ss + delta_H_elast +
                                              self.delta_H_struct(elements, fracs, lattice[i]))
                else:
                    delta_H_ss_list = list()
                    for lattice_possible in ['default', 'fcc', 'bcc', 'hcp']:
                        delta_H_ss_list.append(delta_H_chem_ss + delta_H_elast +
                                               self.delta_H_struct(elements, fracs, lattice_possible))
                    delta_H_ss_min = min(delta_H_ss_list)
                    miedema_result.append(delta_H_ss_min)

            # amor: amorphous phase
            elif struct_now == 'amor':
                delta_H_chem_amor = self.delta_H_chem(elements, fracs, 'amor')
                delta_H_amor = delta_H_chem_amor + 3.5 * np.dot(fracs, melting_point) / 1000
                miedema_result.append(delta_H_amor)

        # convert kJ/mol to eV/atom. The original Miedema model is in kJ/mol.
        miedema_result = [delta_H / 96.4853 for delta_H in miedema_result]
        return miedema_result

    def feature_labels(self):
        labels = []
        for target_struct in self.struct.split('|'):
            if target_struct.startswith('ss'):
                if target_struct != 'ss':
                    for label in target_struct.split(',')[1:]:
                        labels.append('formation_enthalpy_ss_'+label)
                else:
                    labels.append('formation_enthalpy_ss_min')
            else:
                labels.append('formation_enthalpy_'+target_struct)
        return labels


    def citations(self):
        citation = ('@article{de1988cohesion, '
                    'title={Cohesion in metals},'
                    'author={De Boer, Frank R and Mattens, WCM '
                    'and Boom, R and Miedema, AR and Niessen, AK},'
                    'year={1988}}')
        return citation

    def implementors(self):
        return ['Qi Wang']
