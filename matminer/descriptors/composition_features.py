from pymatgen import Element, Composition, MPRester
import collections
import os
import json
import itertools

import numpy as np
import pandas as pd

from base_classes import BaseFeaturizer
from data import DemlData, MagpieData, PymatgenData
from stats import PropertyStats

__author__ = 'Saurabh Bajaj <sbajaj@lbl.gov>, Logan Ward, Jiming Chen, Ashwin Aggarwal, Kiran Mathew'

# TODO: read Magpie file only once
# TODO: Handle dictionaries in case of atomic radii. Aj says "You can require that getting the ionic_radii descriptor
#  requires a valence-decorated Structure or valence-decorated Composition. Otherwise it does not work, i.e. returns
# None. Other radii (e.g. covalent) won't require an oxidation state and people can and should use those for
# non-ionic structures. You can also have a function that returns a mean of ionic_radii for all valences but that
# should not be the default."
# TODO: unit tests
# TODO: most of this code needs to be rewritten ... AJ


# Load elemental cohesive energy data from json file
#with open(os.path.join(os.path.dirname(__file__), 'cohesive_energies.json'), 'r') as f:
#    ce_data = json.load(f)

#empty dictionary for magpie properties
magpie_props = {}

#list of elements
atomic_syms = []
for atomic_no in range(1,104):
    atomic_syms.append(Element.from_Z(atomic_no).symbol)

class StoichAttributes(BaseFeaturizer):
    """
    Class to calculate stoichiometric attributes.

    Parameters:
        p_list (list of ints): list of norms to calculate
    """

    def __init__(self, p_list=None):
        BaseFeaturizer.__init__(self)
        if p_list == None:
            self.p_list = [0,2,3,5,7,10]
        else:
            self.p_list = p_list
 
    def featurize(self, comp_obj):
        """
        Get stoichiometric attributes
        Args:
            comp_obj: Pymatgen composition object
            p_list (list of ints)
        
        Returns: 
            p_norm (float): Lp norm-based stoichiometric attribute
        """
 
        el_amt = comp_obj.get_el_amt_dict()
        
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

        return p_norms

    def generate_labels(self):
        labels = []
        for p in self.p_list:
            labels.append("%d-norm"%p)
        return labels

class ElemPropertyAttributes(BaseFeaturizer):
    """
    Class to calculate elemental property attributes

    Parameters:
        attributes (list of strings): List of elemental properties to use    
    """

    def __init__(self, method="magpie", stats=None, attributes=None, data_source=None):
        BaseFeaturizer.__init__(self)
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

    def featurize(self, comp_obj):
        """
        Get elemental property attributes

        Args:
            comp_obj: Pymatgen composition object
        
        Returns:
            all_attributes: min, max, range, mean, average deviation, and mode of descriptors
        """

        el_amt = comp_obj.get_el_amt_dict()
        elements = list(el_amt.keys())
       
        all_attributes = []

        for attr in self.attributes:

            elem_data = self.data_source.get_property(comp_obj, attr)

            for stat in self.stats:
                all_attributes.append(PropertyStats().calc_stat(elem_data, stat))

        return all_attributes

    def generate_labels(self):
        labels = []
        for attr in self.attributes:
            for stat in self.stats:
                labels.append("%s %s"%(stat, attr))

        return labels

class ValenceOrbitalAttributes(BaseFeaturizer):
    """Class to calculate valence orbital attributes"""

    def __init__(self, data_source=MagpieData()):
        BaseFeaturizer.__init__(self)
        self.data_source = data_source

    def featurize(self, comp_obj):
        """Weighted fraction of valence electrons in each orbital

           Args: 
                comp_obj: Pymatgen composition object

           Returns: 
                Fs, Fp, Fd, Ff (float): Fraction of valence electrons in s, p, d, and f orbitals
        """    
        
        num_atoms = comp_obj.num_atoms

        avg_total_valence = sum(self.data_source.get_property(comp_obj,"NValance"))/num_atoms
        avg_s = sum(self.data_source.get_property(comp_obj,"NsValence"))/num_atoms
        avg_p = sum(self.data_source.get_property(comp_obj,"NpValence"))/num_atoms
        avg_d = sum(self.data_source.get_property(comp_obj,"NdValence"))/num_atoms
        avg_f = sum(self.data_source.get_property(comp_obj,"NfValence"))/num_atoms

        Fs = avg_s/avg_total_valence
        Fp = avg_p/avg_total_valence
        Fd = avg_d/avg_total_valence
        Ff = avg_f/avg_total_valence

        return list((Fs, Fp, Fd, Ff))

    def generate_labels(self):
        orbitals = ["s","p","d","f"]
        labels = []
        for orb in orbitals:
            labels.append("Frac %s Valence Electrons"%orb)

        return labels

class IonicAttributes(BaseFeaturizer):
    """Class to calculate ionic property attributes"""
    def __init__(self, data_source=MagpieData()):
        BaseFeaturizer.__init__(self)
        self.data_source = data_source

    def featurize(self, comp_obj):
        """
        Ionic character attributes

        Args:
            comp_obj: Pymatgen composition object

        Returns:
            cpd_possible (bool): Indicates if a neutral ionic compound is possible
            max_ionic_char (float): Maximum ionic character between two atoms
            avg_ionic_char (float): Average ionic character
        """

        el_amt = comp_obj.get_el_amt_dict()
        elements = list(el_amt.keys())
        values = list(el_amt.values())

        if len(elements) < 2: #Single element
            cpd_possible = True
            max_ionic_char = 0
            avg_ionic_char = 0        
        else:
            #Get magpie data for each element
            all_ox_states = self.data_source.get_property(comp_obj,"OxidationStates")
            all_elec = self.data_source.get_property(comp_obj,"Electronegativity")
            ox_states = []
            elec = []
            
            values_int = map(int, values)
            for i in range(1,len(values_int)+1):
                ind = int(sum(values_int[:i])-1)
                ox_states.append(all_ox_states[ind])
                elec.append(all_elec[ind])

            #Determine if neutral compound is possible
            cpd_possible = False
            ox_sets = itertools.product(*ox_states)
            for ox in ox_sets:
                if np.dot(ox, values) == 0:
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
                ionic_char.append(1.0 - np.exp(-0.25*(XA-XB)**2))
                avg_ionic_char += el_frac[pair[0]]*el_frac[pair[1]]*ionic_char[-1]
            
            max_ionic_char = np.max(ionic_char)
         
        return list((cpd_possible, max_ionic_char, avg_ionic_char))

    def generate_labels(self):
        labels = ["compound possible", "Max Ionic Char", "Avg Ionic Char"]
        return labels
      
class ElementFractionAttribute(BaseFeaturizer):
    """
    Class to calculate the atomic fraction of each element in a composition.

    Generates: vector where each index represents an element in atomic number order. 
    """

    def __init__(self):
        pass

    def featurize(self, comp):
        vector = [0]*103
        el_list = list(comp.element_composition.fractional_composition.items())
        for el in el_list:
            obj = el
            atomic_number_i = obj[0].number-1
            vector[atomic_number_i] = obj[1]
        return vector

    def generate_labels(self):
        labels = []
        for i in range(1, 104):
            labels.append(Element.from_Z(i).symbol)
        return labels

class TMetalFractionAttribute(BaseFeaturizer):
    """
    Class to calculate fraction of magnetic transition metals in a composition.

    Generates: Fraction of magnetic transition metal atoms in a compound
    """

    def __init__(self):
        self.magn_elem = ['Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Nb','Mo','Tc','Ru',
            'Rh','Pd','Ag','Ta','W','Re','Os','Ir','Pt']

    def featurize(self, comp):
        el_amt = comp.get_el_amt_dict()

        frac_magn_atoms = 0
        for el in el_amt:
            if el in self.magn_elem:
                frac_magn_atoms += el_amt[el]

        frac_magn_atoms /= sum(el_amt.values())

        return frac_magn_atoms

    def generate_labels(self):
        labels = ["TMetal Fraction"]
        return labels

class ElectronAffinityAttribute(BaseFeaturizer):
    """
    Class to calculate average electron affinity times formal charge of anion elements

    Generates average (electron affinity*formal charge) of anions
    """

    def __init__(self, data_source=DemlData()):
        self.data_source = data_source

    def featurize(self, comp):
        fml_charge = self.data_source.get_property(comp, "formal_charge")
        electron_affin = self.data_source.get_property(comp, "electron_affin")

        anion_charge = []
        anion_affin = []

        for i in range(len(fml_charge)):
            if fml_charge[i] < 0:
                anion_charge.append(fml_charge[i])
                anion_affin.append(electron_affin[i])

        avg_anion_affin = np.dot(anion_charge, anion_affin)/len(fml_charge)

        return avg_anion_affin

    def generate_labels(self):
        labels = ["Avg Anion Electron Affinity"]
        return labels

class ElectronegativityDiffAttribute(BaseFeaturizer):
    """
    Class to calculate average electronegativity difference

    Generates average electronegativity difference between cations and anions
    """

    def __init__(self, data_source=DemlData(), stats=None):
        self.data_source = data_source
        if stats == None:
            self.stats = ["minimum","maximum","range","mean","std_dev"]
        else:
            self.stats = stats

    def featurize(self, comp):

        fml_charge = self.data_source.get_property(comp, "formal_charge")
        electroneg = self.data_source.get_property(comp, "electronegativity")
        
        cation_en = []
        anion_en = []

        #Get electronegativity values for cations and anions
        for i in range(len(fml_charge)):
            if fml_charge[i] > 0:
                cation_en.append(electroneg[i])
            else:
                anion_en.append(electroneg[i])

        avg_en_diff = []
        n_anions = len(anion_en)
        for en in cation_en:
            avg_en_diff.append(abs(en - (sum(anion_en)/n_anions)))

        en_diff_stats = []

        for stat in self.stats:
            en_diff_stats.append(PropertyStats().calc_stat(avg_en_diff, stat))

        if "mean" in self.stats: #Change to mean across total atoms
            en_diff_stats[self.stats.index("mean")] *= float(len(cation_en))/float(len(fml_charge))

        return en_diff_stats

    def generate_labels(self):

        labels = []
        for stat in self.stats:
            labels.append("%s EN difference"%stat)

        return labels

#Old functions below
def get_pymatgen_descriptor(comp, prop):
    """
    Get descriptor data for elements in a compound from pymatgen.

    Args:
        comp: (str) compound composition, eg: "NaCl"
        prop: (str) pymatgen element attribute, as defined in the Element class at
            http://pymatgen.org/_modules/pymatgen/core/periodic_table.html

    Returns: (list) of values containing descriptor floats for each atom in the compound

    """
    eldata = []
    eldata_tup_lst = []
    eldata_tup = collections.namedtuple('eldata_tup', 'element propname propvalue propunit amt')
    el_amt_dict = Composition(comp).get_el_amt_dict()

    for el in el_amt_dict:

        if callable(getattr(Element(el), prop)) is None:
            raise ValueError('Invalid pymatgen Element attribute(property)')

        if getattr(Element(el), prop) is not None:

            # units are None for these pymatgen descriptors
            # todo: there seem to be a lot more unitless descriptors which are not listed here... -Alex D
            if prop in ['X', 'Z', 'ionic_radii', 'group', 'row', 'number', 'mendeleev_no','oxidation_states','common_oxidation_states']:
                units = None
            else:
                units = getattr(Element(el), prop).unit

            # Make a named tuple out of all the available information
            eldata_tup_lst.append(
                eldata_tup(element=el, propname=prop, propvalue=float(getattr(Element(el), prop)), propunit=units,
                           amt=el_amt_dict[el]))

            # Add descriptor values, one for each atom in the compound
            for i in range(int(el_amt_dict[el])):
                eldata.append(float(getattr(Element(el), prop)))

        else:
            eldata_tup_lst.append(eldata_tup(element=el, propname=prop, propvalue=None, propunit=None,
                                             amt=el_amt_dict[el]))

    return eldata


def get_magpie_descriptor(comp, descriptor_name):
    """
    Get descriptor data for elements in a compound from the Magpie data repository.

    Args:
        comp: (str) compound composition, eg: "NaCl"
        descriptor_name: name of Magpie descriptor needed. Find the entire list at
            https://bitbucket.org/wolverton/magpie/src/6ecf8d3b79e03e06ef55c141c350a08fbc8da849/Lookup%20Data/?at=master

    Returns: (list) of descriptor values for each atom in the composition

    """
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_files", 'magpie_elementdata')
    magpiedata = []
    magpiedata_tup_lst = []
    magpiedata_tup = collections.namedtuple('magpiedata_tup', 'element propname propvalue propunit amt')
    available_props = []

    # Make a list of available properties

    for datafile in os.listdir(data_dir):
        available_props.append(datafile.replace('.table', ''))

    if descriptor_name not in available_props:
        raise ValueError(
            "This descriptor is not available from the Magpie repository. Choose from {}".format(available_props))

    # Get units from Magpie README file
    el_amt = Composition(comp).get_el_amt_dict()
    unit = None
    with open(os.path.join(data_dir, 'README.txt'), 'r') as readme_file:
        readme_file_line = readme_file.readlines()
        for lineno, line in enumerate(readme_file_line, 1):
            if descriptor_name + '.table' in line:
                if 'Units: ' in readme_file_line[lineno + 1]:
                    unit = readme_file_line[lineno + 1].split(':')[1].strip('\n')

    # Extract from data file
    with open(os.path.join(data_dir, '{}.table'.format(descriptor_name)), 'r') as descp_file:
        lines = descp_file.readlines()
        for el in el_amt:
            atomic_no = Element(el).Z
            #if len(lines[atomic_no - 1].split()) > 1:
            if descriptor_name in ["OxidationStates"]:
                propvalue = [float(i) for i in lines[atomic_no - 1].split()]
            else:
                propvalue = float(lines[atomic_no - 1])

            magpiedata_tup_lst.append(magpiedata_tup(element=el, propname=descriptor_name,
                                                    propvalue=propvalue, propunit=unit,
                                                    amt=el_amt[el]))

            # Add descriptor values, one for each atom in the compound
            for i in range(int(el_amt[el])):
                magpiedata.append(propvalue)

    return magpiedata

def get_cohesive_energy(comp):
    """
    Get cohesive energy of compound by subtracting elemental cohesive energies from the formation energy of the compund.
    Elemental cohesive energies are taken from http://www.      knowledgedoor.com/2/elements_handbook/cohesive_energy.html.
    Most of them are taken from "Charles Kittel: Introduction to Solid State Physics, 8th edition. Hoboken, NJ:
    John Wiley & Sons, Inc, 2005, p. 50."

    Args:
        comp: (str) compound composition, eg: "NaCl"

    Returns: (float) cohesive energy of compound

    """
    el_amt_dict = Composition(comp).get_el_amt_dict()

    # Get formation energy of most stable structure from MP
    struct_lst = MPRester().get_data(comp)
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

    return cohesive_energy

def band_center(comp):
    """
    Estimate absolution position of band center using geometric mean of electronegativity
    Ref: Butler, M. a. & Ginley, D. S. Prediction of Flatband Potentials at Semiconductor-Electrolyte Interfaces from
    Atomic Electronegativities. J. Electrochem. Soc. 125, 228 (1978).

    Args:
        comp: (Composition)

    Returns: (float) band center

    """
    prod = 1.0
    for el, amt in comp.get_el_amt_dict().iteritems():
        prod = prod * (Element(el).X ** amt)

    return -prod ** (1 / sum(comp.get_el_amt_dict().values()))


def get_holder_mean(data_lst, power):
    """
    Get Holder mean

    Args:
        data_lst: (list/array) of values
        power: (int/float) non-zero real number

    Returns: Holder mean

    """
    # Function for calculating Geometric mean
    geomean = lambda n: reduce(lambda x, y: x * y, n) ** (1.0 / len(n))

    # If power=0, return geometric mean
    if power == 0:
        return geomean(data_lst)

    else:
        total = 0.0
        for value in data_lst:
            total += value ** power
        return (total / len(data_lst)) ** (1 / float(power))

if __name__ == '__main__':
    descriptors = ['atomic_mass', 'X', 'Z', 'thermal_conductivity', 'melting_point',
                   'coefficient_of_linear_thermal_expansion']

    for desc in descriptors:
        print(get_pymatgen_descriptor('LiFePO4', desc))
    print(get_magpie_descriptor('LiFePO4', 'AtomicVolume'))
    print(get_magpie_descriptor('LiFePO4', 'Density'))
    print(get_holder_mean([1, 2, 3, 4], 0))
   
    training_set = pd.DataFrame({"composition":["Fe2O3"]})
    print "WARD NPJ ATTRIBUTES"
    print "Stoichiometric attributes"
    p_list = [0,2,3,5,7,9]
    print StoichAttributes().featurize_all(training_set)
    print "Elemental property attributes"
    print ElemPropertyAttributes().featurize_all(training_set)
    print "Valence Orbital Attributes"
    print ValenceOrbitalAttributes().featurize_all(training_set)
    print "Ionic attributes"
    print IonicAttributes().featurize_all(training_set)

    print "DEML ELEMENTAL DESCRIPTORS"
    print ElemPropertyAttributes(method="deml").featurize_all(training_set)
    print TMetalFractionAttribute().featurize_all(training_set)
    print ElectronAffinityAttribute().featurize_all(training_set)
