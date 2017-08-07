from pymatgen import Element, Composition, MPRester
from pymatgen.core.periodic_table import get_el_sp

import collections
import os
import json
import itertools

import numpy as np
import pandas as pd

from matminer.featurizers.base import BaseFeaturizer
from matminer.featurizers.data import DemlData, MagpieData, PymatgenData
from matminer.featurizers.stats import PropertyStats

__author__ = 'Saurabh Bajaj <sbajaj@lbl.gov>, Logan Ward, Jiming Chen, Ashwin Aggarwal, Kiran Mathew, Anubhav Jain'

# TODO: read Magpie file only once
# TODO: Handle dictionaries in case of atomic radii. Aj says "You can require that getting the ionic_radii descriptor
#  requires a valence-decorated Structure or valence-decorated Composition. Otherwise it does not work, i.e. returns
# None. Other radii (e.g. covalent) won't require an oxidation state and people can and should use those for
# non-ionic structures. You can also have a function that returns a mean of ionic_radii for all valences but that
# should not be the default."
# TODO: unit tests
# TODO: most of this code needs to be rewritten ... AJ

module_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(module_dir, 'data_files', 'cohesive_energies.json'), 'r') as f:
    ce_data = json.load(f)

#empty dictionary for magpie properties
magpie_props = {}

#list of elements
atomic_syms = []
for atomic_no in range(1,104):
    atomic_syms.append(Element.from_Z(atomic_no).symbol)

class Stoichiometry(BaseFeaturizer):
    """
    Class to calculate stoichiometric attributes.

    Parameters:
        p_list (list of ints): list of norms to calculate
        num_atoms (bool): whether to return number of atoms
    """

    def __init__(self, p_list=[0,2,3,5,7,10], num_atoms=False):
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
            stoich_attr = [n_atoms] #return number of atoms if no norms specified
        else: 
            p_norms = [0]*len(self.p_list)
            n_atoms = sum(el_amt.values())

            for i in range(len(self.p_list)):
                if self.p_list[i] < 0:
                    raise ValueError("p-norm not defined for p < 0")
                if self.p_list[i] == 0:
                    p_norms[i] = len(el_amt.values())
                else:
                    for j in el_amt:
                        p_norms[i] += (el_amt[j]/n_atoms)**self.p_list[i]
                    p_norms[i] = p_norms[i]**(1.0/self.p_list[i])

            if self.num_atoms:
                stoich_attr = [n_atoms] + p_norms
            else:
                stoich_attr = p_norms

        return stoich_attr

    def feature_labels(self):
        labels = []
        if self.num_atoms:
            labels.append("Number of atoms")

        if self.p_list != None:
            for p in self.p_list:
                labels.append("%d-norm"%p)

        return labels

    def citations(self):
        citation = ("@article{ward_agrawal_choudary_wolverton_2016, title={A general-purpose "
            "machine learning framework for predicting properties of inorganic materials}, "
            "volume={2}, DOI={10.1038/npjcompumats.2017.28}, number={1}, journal={npj "
            "Computational Materials}, author={Ward, Logan and Agrawal, Ankit and Choudhary, "
            "Alok and Wolverton, Christopher}, year={2016}}")
        return citation

    def implementors(self):
        return ["Jiming Chen", "Logan Ward"]

class ElementProperty(BaseFeaturizer):
    """
    Class to calculate elemental property attributes

    Parameters:
        attributes (list of strings): List of elemental properties to use
        method (string): pre-packaged sets of property sets to compute
        data_source (data object): source from which to retrieve element property data
    """

    def __init__(self, method="magpie", stats=None, attributes=None, data_source=None):
        self.method = method
        if self.method == "deml":
            self.stats = ["minimum", "maximum", "range", "mean", "std_dev"]
            self.attributes = ["atom_num", "atom_mass", "row_num", "col_num", "atom_radius", "molar_vol", "heat_fusion", "melting_point",
                "boiling_point", "heat_cap", "first_ioniz", "total_ioniz", "electronegativity", "formal_charge", "xtal_field_split",
                "magn_moment", "so_coupling", "sat_magn", "electric_pol", "GGAU_Etot", "mus_fere"]
            self.data_source = DemlData()
        elif self.method == "magpie":
            self.stats = ["minimum", "maximum", "range", "mean", "avg_dev", "mode"]
            self.attributes = ["Number", "MendeleevNumber", "AtomicWeight","MeltingT","Column","Row","CovalentRadius","Electronegativity",
                "NsValence","NpValence","NdValence","NfValence","NValance","NsUnfilled","NpUnfilled","NdUnfilled","NfUnfilled","NUnfilled",
                "GSvolume_pa","GSbandgap","GSmagmom","SpaceGroupNumber"]
            self.data_source = MagpieData()
        else:
            self.stats = stats
            self.attributes = attributes
            self.data_source = data_source

    def featurize(self, comp):
        """
        Get elemental property attributes

        Args:
            comp: Pymatgen composition object

        Returns:
            all_attributes: Specified property statistics of descriptors
        """

        el_amt = comp.fractional_composition.get_el_amt_dict()
        elements = sorted(el_amt.keys(), key=lambda sym: get_el_sp(sym).X)
        fracs = [el_amt[el] for el in elements]

        all_attributes = []

        for attr in self.attributes:
            elem_data = self.data_source.get_property(comp, attr, return_per_element=True)

            for stat in self.stats:
                all_attributes.append(PropertyStats().calc_stat(stat, elem_data, weights=fracs))

        return all_attributes

    def feature_labels(self):
        labels = []
        for attr in self.attributes:
            for stat in self.stats:
                labels.append("%s %s"%(stat, attr))

        return labels

    def citations(self):
        if self.method == "magpie":
            citation = ("@article{ward_agrawal_choudary_wolverton_2016, title={A general-purpose "
                "machine learning framework for predicting properties of inorganic materials}, "
                "volume={2}, DOI={10.1038/npjcompumats.2017.28}, number={1}, journal={npj "
                "Computational Materials}, author={Ward, Logan and Agrawal, Ankit and Choudhary, "
                "Alok and Wolverton, Christopher}, year={2016}}")
        elif self.method == "deml":
            citation = ("@article{deml_ohayre_wolverton_stevanovic_2016, title={Predicting density "
                "functional theory total energies and enthalpies of formation of metal-nonmetal "
                "compounds by linear regression}, volume={47}, DOI={10.1002/chin.201644254}, "
                "number={44}, journal={ChemInform}, author={Deml, Ann M. and Ohayre, Ryan and "
                "Wolverton, Chris and Stevanovic, Vladan}, year={2016}}")
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

    def __init__(self, data_source=MagpieData(), orbitals=["s","p","d","f"], props=["avg", "frac"]):
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
        elements = sorted(el_amt.keys(), key=lambda sym: get_el_sp(sym).X)
        el_fracs = [el_amt[el] for el in elements]

        avg = []

        for orb in self.orbitals:
            avg.append(PropertyStats().mean(self.data_source.get_property(comp,"N%sValence"%orb, return_per_element=True), weights=el_fracs))

        if "frac" in self.props:
            avg_total_valence = PropertyStats().mean(self.data_source.get_property(comp,"NValance", return_per_element=True), weights=el_fracs)
            frac = [a/avg_total_valence for a in avg]

        valence_attributes = []
        for prop in self.props:
            valence_attributes += locals()[prop]

        return valence_attributes

    def feature_labels(self):
        labels = []
        for prop in self.props:
            for orb in self.orbitals:
                labels.append("%s %s valence electrons"%(prop, orb))

        return labels

    def citations(self):
        ward_citation = ("@article{ward_agrawal_choudary_wolverton_2016, title={A general-purpose "
            "machine learning framework for predicting properties of inorganic materials}, "
            "volume={2}, DOI={10.1038/npjcompumats.2017.28}, number={1}, journal={npj "
            "Computational Materials}, author={Ward, Logan and Agrawal, Ankit and Choudhary, "
            "Alok and Wolverton, Christopher}, year={2016}}")
        deml_citation = ("@article{deml_ohayre_wolverton_stevanovic_2016, title={Predicting density "
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
        elements = sorted(el_amt.keys(), key=lambda sym: get_el_sp(sym).X)
        values = [el_amt[el] for el in elements]

        if len(elements) < 2: #Single element
            cpd_possible = True
            max_ionic_char = 0
            avg_ionic_char = 0
        else:
            #Get magpie data for each element
            ox_states = self.data_source.get_property(comp, "OxidationStates", return_per_element=True)
            elec = self.data_source.get_property(comp, "Electronegativity", return_per_element=True)

            #Determine if neutral compound is possible
            cpd_possible = False
            ox_sets = itertools.product(*ox_states)
            for ox in ox_sets:
                if abs(np.dot(ox, values)) < 1e-4:
                    cpd_possible = True
                    break

            #Ionic character attributes
            atom_pairs = itertools.combinations(range(len(elements)), 2)
            el_frac = list(np.divide(values, sum(values)))

            ionic_char = []
            avg_ionic_char = 0

            for pair in atom_pairs:
                XA = elec[pair[0]]
                XB = elec[pair[1]]
                ionic_char.append(1.0 - np.exp(-0.25*((XA-XB)**2)))
                avg_ionic_char += el_frac[pair[0]]*el_frac[pair[1]]*ionic_char[-1]

            max_ionic_char = np.max(ionic_char)

        return list((cpd_possible, max_ionic_char, avg_ionic_char))

    def feature_labels(self):
        labels = ["compound possible", "Max Ionic Char", "Avg Ionic Char"]
        return labels

    def citations(self):
        citation = ("@article{ward_agrawal_choudary_wolverton_2016, title={A general-purpose "
            "machine learning framework for predicting properties of inorganic materials}, "
            "volume={2}, DOI={10.1038/npjcompumats.2017.28}, number={1}, journal={npj "
            "Computational Materials}, author={Ward, Logan and Agrawal, Ankit and Choudhary, "
            "Alok and Wolverton, Christopher}, year={2016}}")
        return citation

    def implementors(self):
        return ["Jiming Chen", "Logan Ward"]

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

        vector = [0]*103
        el_list = list(comp.element_composition.fractional_composition.items())
        for el in el_list:
            obj = el
            atomic_number_i = obj[0].number-1
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
        self.magn_elem = ['Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Nb','Mo','Tc','Ru',
            'Rh','Pd','Ag','Ta','W','Re','Os','Ir','Pt']

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
        labels = ["TMetal Fraction"]
        return labels

    def citations(self):
        citation = ("@article{deml_ohayre_wolverton_stevanovic_2016, title={Predicting density "
            "functional theory total energies and enthalpies of formation of metal-nonmetal "
            "compounds by linear regression}, volume={47}, DOI={10.1002/chin.201644254}, "
            "number={44}, journal={ChemInform}, author={Deml, Ann M. and Ohayre, Ryan and "
            "Wolverton, Chris and Stevanovic, Vladan}, year={2016}}")
        return citation

    def implementors(self):
        return ["Jiming Chen, Logan Ward"]

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

        el_amt = comp.fractional_composition.get_el_amt_dict()
        elements = sorted(el_amt.keys(), key=lambda sym: get_el_sp(sym).X)
        el_fracs = [el_amt[el] for el in elements]

        fml_charge = self.data_source.get_property(comp, "formal_charge", return_per_element=True)
        electron_affin = self.data_source.get_property(comp, "electron_affin", return_per_element=True)

        anion_charge = []
        anion_affin = []

        avg_anion_affin = 0

        for i in range(len(fml_charge)):
            if fml_charge[i] < 0:
                avg_anion_affin += fml_charge[i]*electron_affin[i]*el_fracs[i]

        return [avg_anion_affin]

    def feature_labels(self):
        labels = ["Avg Anion Electron Affinity"]
        return labels

    def citations(self):
        citation = ("@article{deml_ohayre_wolverton_stevanovic_2016, title={Predicting density "
            "functional theory total energies and enthalpies of formation of metal-nonmetal "
            "compounds by linear regression}, volume={47}, DOI={10.1002/chin.201644254}, "
            "number={44}, journal={ChemInform}, author={Deml, Ann M. and Ohayre, Ryan and "
            "Wolverton, Chris and Stevanovic, Vladan}, year={2016}}")
        return citation

    def implementors(self):
        return ["Jiming Chen", "Logan Ward"]

class ElectronegativityDiff(BaseFeaturizer):
    """
    Class to calculate average electronegativity difference

    Parameters:
        data_source (data class): source from which to retrieve element data
        stats: Property statistics to compute

    Generates average electronegativity difference between cations and anions
    """

    def __init__(self, data_source=DemlData(), stats=None):
        self.data_source = data_source
        if stats == None:
            self.stats = ["minimum","maximum","range","mean","std_dev"]
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
        elements = sorted(el_amt.keys(), key=lambda sym: get_el_sp(sym).X)

        fml_charge = self.data_source.get_property(comp, "formal_charge", return_per_element=True)
        electroneg = self.data_source.get_property(comp, "electronegativity", return_per_element=True)

        cations = []
        anions = []
        cation_en = []
        anion_en = []

        #Get electronegativity values for cations and anions
        for i in range(len(fml_charge)):
            if fml_charge[i] > 0:
                cations.append(elements[i])
                cation_en.append(electroneg[i])
            elif fml_charge[i] < 0:
                anions.append(elements[i])
                anion_en.append(electroneg[i])

        if len(cations) == 0 or len(anions) == 0: #Return NaN if cations/anions missing
            return len(self.stats)*[float("NaN")]

        avg_en_diff = []
        n_anions = sum([el_amt[el] for el in anions])

        for cat_en in cation_en:
            en_diff = 0
            for i in range(len(anions)):
                frac_anion = el_amt[anions[i]]/n_anions
                an_en = anion_en[i]
                en_diff += abs(cat_en - an_en)*frac_anion
            avg_en_diff.append(en_diff)

        cation_fracs = [el_amt[el] for el in cations]
        en_diff_stats = []

        for stat in self.stats:
            if stat == "std_dev":
                en_diff_stats.append(PropertyStats().calc_stat(stat, avg_en_diff))
            else:
                en_diff_stats.append(PropertyStats().calc_stat(stat, avg_en_diff, weights=cation_fracs))

        return en_diff_stats

    def feature_labels(self):

        labels = []
        for stat in self.stats:
            labels.append("%s EN difference"%stat)

        return labels

    def citations(self):
        citation = ("@article{deml_ohayre_wolverton_stevanovic_2016, title={Predicting density "
            "functional theory total energies and enthalpies of formation of metal-nonmetal "
            "compounds by linear regression}, volume={47}, DOI={10.1002/chin.201644254}, "
            "number={44}, journal={ChemInform}, author={Deml, Ann M. and Ohayre, Ryan and "
            "Wolverton, Chris and Stevanovic, Vladan}, year={2016}}")
        return citation

    def implementors(self):
        return ["Jiming Chen", "Logan Ward"]

class FERECorrection(BaseFeaturizer):
    """
    Class to calculate difference between fitted elemental-phase reference energy (FERE) and GGA+U energy

    Parameters:
        data_source (data class): source from which to retrieve element data
        stats: Property statistics to compute

    Generates: Property statistics of difference between FERE and GGA+U energy
    """

    def __init__(self, data_source=DemlData(), stats=None):
        self.data_source = data_source
        if stats == None:
            self.stats = ["minimum", "maximum","range","mean","std_dev"]
        else:
            self.stats = stats

    def featurize(self, comp):
        """
        Args:
            comp: Pymatgen Composition object

        Returns:
            fere_corr_stats (list of floats): Property stats of FERE Correction
        """

        el_amt = comp.fractional_composition.get_el_amt_dict()
        elements = sorted(el_amt.keys(), key=lambda sym: get_el_sp(sym).X)
        el_frac = [el_amt[el] for el in elements]

        GGAU_Etot = self.data_source.get_property(comp, "GGAU_Etot", return_per_element=True)
        mus_fere = self.data_source.get_property(comp, "mus_fere", return_per_element=True)

        fere_corr = [mus_fere[i] - GGAU_Etot[i] for i in range(len(GGAU_Etot))]

        fere_corr_stats = []
        for stat in self.stats:
            fere_corr_stats.append(PropertyStats().calc_stat(stat, fere_corr, weights=el_frac))

        return fere_corr_stats

    def feature_labels(self):

        labels = []
        for stat in self.stats:
            labels.append("%s FERE Correction"%stat)

        return labels

    def citations(self):
        citation = ("@article{deml_ohayre_wolverton_stevanovic_2016, title={Predicting density "
            "functional theory total energies and enthalpies of formation of metal-nonmetal "
            "compounds by linear regression}, volume={47}, DOI={10.1002/chin.201644254}, "
            "number={44}, journal={ChemInform}, author={Deml, Ann M. and Ohayre, Ryan and "
            "Wolverton, Chris and Stevanovic, Vladan}, year={2016}}")
        return citation

    def implementors(self):
        return ["Jiming Chen", "Logan Ward"]

class CohesiveEnergy(BaseFeaturizer):

    def featurize(self, comp):
        """
        Get cohesive energy of compound by subtracting elemental cohesive energies from the formation energy of the compund.
        Elemental cohesive energies are taken from http://www.      knowledgedoor.com/2/elements_handbook/cohesive_energy.html.
        Most of them are taken from "Charles Kittel: Introduction to Solid State Physics, 8th edition. Hoboken, NJ:
        John Wiley & Sons, Inc, 2005, p. 50."

        Args:
            comp: (str) compound composition, eg: "NaCl"

        Returns: (float) cohesive energy of compound

        """
        el_amt_dict = comp.get_el_amt_dict()

        # Get formation energy of most stable structure from MP
        struct_lst = MPRester().get_data(comp.formula.replace(" ",""))
        if len(struct_lst) > 0:
            struct_lst = sorted(struct_lst, key=lambda e: e['energy_per_atom'])
            most_stable_entry = struct_lst[0]
            formation_energy = most_stable_entry['formation_energy_per_atom']
        else:
            raise ValueError('No structure found in MP for {}'.format(comp))

        # Subtract elemental cohesive energies from formation energy
        cohesive_energy = formation_energy
        for el in el_amt_dict:
            cohesive_energy -= el_amt_dict[el] * ce_data[el]

        return [cohesive_energy]

    def feature_labels(self):
        return ["Cohesive Energy"]

    def implementors(self):
        return ["Saurabh Bajaj"]

class BandCenter(BaseFeaturizer):

    def featurize(self, comp):
        """
        Estimate absolution position of band center using geometric mean of electronegativity
        Ref: Butler, M. a. & Ginley, D. S. Prediction of Flatband Potentials at Semiconductor-Electrolyte Interfaces from
        Atomic Electronegativities. J. Electrochem. Soc. 125, 228 (1978).

        Args:
            comp: (Composition)

        Returns: (float) band center

        """
        prod = 1.0
        for el, amt in comp.get_el_amt_dict().items():
            prod = prod * (Element(el).X ** amt)

        return [-prod ** (1 / sum(comp.get_el_amt_dict().values()))]

    def feature_labels(self):
        return ["Band Center"]

    def implementors(self):
        return ["Anubhav Jain"]

if __name__ == '__main__':
    print(PropertyStats.holder_mean([1, 2, 3, 4]))
   
    training_set = pd.DataFrame({"composition":[Composition("Fe2O3"), Composition("Ga1Na6P3"), Composition("O4Si1Zn2")]})
    print("WARD NPJ ATTRIBUTES")
    print("Stoichiometric attributes")
    p_list = [0,2,3,5,7,9]
    print(Stoichiometry().featurize_dataframe(training_set, col_id="composition"))
    print("Elemental property attributes")
    print(ElementProperty().featurize_dataframe(training_set, col_id="composition"))
    print("Valence Orbital Attributes")
    print(ValenceOrbital(props=["frac"]).featurize_dataframe(training_set, col_id="composition"))
    print("Ionic attributes")
    print(IonProperty().featurize_dataframe(training_set, col_id="composition"))

    print("DEML ELEMENTAL DESCRIPTORS")
    print(Stoichiometry(p_list=None, num_atoms=True).featurize_dataframe(training_set, col_id="composition"))
    print(ElementProperty(method="deml").featurize_dataframe(training_set, col_id="composition"))
    print(TMetalFraction().featurize_dataframe(training_set, col_id="composition"))
    print(ElectronAffinity().featurize_dataframe(training_set, col_id="composition"))
    print(ValenceOrbital(orbitals=["s", "p", "d"], props=["avg", "frac"]).featurize_dataframe(training_set, col_id="composition"))
    print(ElectronegativityDiff().featurize_dataframe(training_set, col_id="composition"))
