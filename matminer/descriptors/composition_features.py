from __future__ import division, unicode_literals, print_function

import collections
import json
import os
import re
from functools import reduce
import itertools

import numpy as np

from monty.design_patterns import singleton

from pymatgen import Element, Composition, MPRester
from pymatgen.core.units import Unit
from pymatgen.core.periodic_table import get_el_sp, _pt_data

from matminer.descriptors.base import BaseFeaturizer

__author__ = 'Jimin Chen, Logan Ward, Saurabh Bajaj, Anubhav jain, Kiran Mathew'

# TODO: unit tests

# Load elemental cohesive energy data from json file
with open(os.path.join(os.path.dirname(__file__), 'cohesive_energies.json'), 'r') as f:
    ce_data = json.load(f)


@singleton    
class MagpieData:
    """
    Singleton class to get data from Magpie files
    """

    def __init__(self):
        self.magpie_props = {}
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", 'magpie_elementdata')
        available_props = []
        prop_value = []        

        # Make a list of available properties
        for datafile in os.listdir(data_dir):
            available_props.append(datafile.replace('.table', ''))

        if descriptor_name not in available_props:
            raise ValueError("This descriptor is not available from the Magpie repository. Choose from {}".format(available_props))

        with open(os.path.join(data_dir, '{}.table'.format(descriptor_name)), 'r') as descp_file:
            lines = descp_file.readlines()
            for atomic_no in range(1, len(_pt_data.keys())+1):  # as high as pymatgen goes
                try:
                    if descriptor_name in ["OxidationStates"]:
                        prop_value.append([float(i) for i in lines[atomic_no - 1].split()])
                    else:
                        prop_value.append(float(lines[atomic_no - 1]))
                except:
                    prop_value.append(float("NaN"))

            self.magpie_props[descriptor_name] = dict(zip(_pt_data.keys(), prop_value))

    def get_data(self, comp_obj, descriptor_name):
        """
        Gets magpie data for a composition object.

        Args:
            comp_obj: Pymatgen composition object
            descriptor_name (string): Name of descriptor

        Returns:
            magpiedata (list): list of values for each atom in comp_obj
        """
        # Get data for given element/compound
        el_amt = comp_obj.get_el_amt_dict()
        elements = list(el_amt.keys())

        magpiedata = [self.magpie_props[descriptor_name][el]
                      for el in elements
                      for i in range(int(el_amt[el]))]

        return magpiedata

magpie_data = MagpieData()    


class StoichAttributes(BaseFeaturizer):
    """
    Class to calculate stoichiometric attributes.

    Parameters:
        p_list (list of ints): list of norms to calculate
    """

    def __init__(self, p_list=None):
        if p_list == None:
            self.p_list = [0, 2, 3, 5, 7, 10]
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

        return p_norms

    def generate_labels(self):
        labels = []
        for p in self.p_list:
            labels.append("%d-norm" % p)
        return labels


class ElemPropertyAttributes(BaseFeaturizer):
    """
    Class to calculate elemental property attributes

    Parameters:
        attributes (list of strings): List of elemental properties to use
    """

    def __init__(self, attributes=None):
        if attributes is None:
            self.attributes = ["Number", "MendeleevNumber", "AtomicWeight", "MeltingT", "Column",
                               "Row", "CovalentRadius", "Electronegativity",
                               "NsValence", "NpValence", "NdValence", "NfValence", "NValance",
                               "NsUnfilled", "NpUnfilled", "NdUnfilled", "NfUnfilled", "NUnfilled",
                               "GSvolume_pa", "GSbandgap", "GSmagmom", "SpaceGroupNumber"]
        else:
            self.attributes = attributes

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
            elem_data = magpie_data.get_data(comp_obj, attr)

            all_attributes.append(min(elem_data))
            all_attributes.append(max(elem_data))
            all_attributes.append(max(elem_data) - min(elem_data))

            prop_mean = sum(elem_data) / len(elem_data)
            all_attributes.append(prop_mean)
            all_attributes.append(sum(np.abs(np.subtract(elem_data, prop_mean))) / len(elem_data))
            all_attributes.append(max(set(elem_data), key=elem_data.count))

        return all_attributes

    def generate_labels(self):
        labels = []
        for attr in self.attributes:
            labels.append("Min %s" % attr)
            labels.append("Max %s" % attr)
            labels.append("Range %s" % attr)
            labels.append("Mean %s" % attr)
            labels.append("AbsDev %s" % attr)
            labels.append("Mode %s" % attr)
        return labels


class ValenceOrbitalAttributes(BaseFeaturizer):
    """Class to calculate valence orbital attributes"""

    def __init__(self):
        pass

    def featurize(self, comp_obj):
        """Weighted fraction of valence electrons in each orbital

           Args:
                comp_obj: Pymatgen composition object

           Returns:
                Fs, Fp, Fd, Ff (float): Fraction of valence electrons in s, p, d, and f orbitals
        """

        num_atoms = comp_obj.num_atoms

        avg_total_valence = sum(magpie_data.get_data(comp_obj, "NValance")) / num_atoms
        avg_s = sum(magpie_data.get_data(comp_obj, "NsValence")) / num_atoms
        avg_p = sum(magpie_data.get_data(comp_obj, "NpValence")) / num_atoms
        avg_d = sum(magpie_data.get_data(comp_obj, "NdValence")) / num_atoms
        avg_f = sum(magpie_data.get_data(comp_obj, "NfValence")) / num_atoms

        Fs = avg_s / avg_total_valence
        Fp = avg_p / avg_total_valence
        Fd = avg_d / avg_total_valence
        Ff = avg_f / avg_total_valence

        return list((Fs, Fp, Fd, Ff))

    def generate_labels(self):
        orbitals = ["s", "p", "d", "f"]
        labels = []
        for orb in orbitals:
            labels.append("Frac %s Valence Electrons" % orb)

        return labels


class IonicAttributes(BaseFeaturizer):
    """Class to calculate ionic property attributes"""

    def __init__(self):
        pass

    def featurize(self, comp_obj):
        """
        Ionic character attributes

        Args:
            com_obj: Pymatgen composition object

        Returns:
            cpd_possible (bool): Indicates if a neutral ionic compound is possible
            max_ionic_char (float): Maximum ionic character between two atoms
            avg_ionic_char (float): Average ionic character
        """

        el_amt = comp_obj.get_el_amt_dict()
        elements = list(el_amt.keys())
        values = list(el_amt.values())

        if len(elements) < 2:  # Single element
            cpd_possible = True
            max_ionic_char = 0
            avg_ionic_char = 0
        else:
            # Get magpie data for each element
            all_ox_states = magpie_data.get_data(comp_obj, "OxidationStates")
            all_elec = magpie_data.get_data(comp_obj, "Electronegativity")
            ox_states = []
            elec = []

            for i in range(1, len(values) + 1):
                ind = int(sum(values[:i]) - 1)
                ox_states.append(all_ox_states[ind])
                elec.append(all_elec[ind])

            # Determine if neutral compound is possible
            cpd_possible = False
            ox_sets = itertools.product(*ox_states)
            for ox in ox_sets:
                if np.dot(ox, values) == 0:
                    cpd_possible = True
                    break

                    # Ionic character attributes
            atom_pairs = itertools.combinations(range(len(elements)), 2)
            el_frac = list(np.divide(values, sum(values)))

            ionic_char = []
            avg_ionic_char = 0

            for pair in atom_pairs:
                XA = elec[pair[0]]
                XB = elec[pair[1]]
                ionic_char.append(1.0 - np.exp(-0.25 * (XA - XB) ** 2))
                avg_ionic_char += el_frac[pair[0]] * el_frac[pair[1]] * ionic_char[-1]

            max_ionic_char = np.max(ionic_char)

        return list((cpd_possible, max_ionic_char, avg_ionic_char))

    def generate_labels(self):
        labels = ["compound possible", "Max Ionic Char", "Avg Ionic Char"]
        return labels


def get_pymatgen_descriptor(composition, property_name):
    """
    Get descriptor data for elements in a compound from pymatgen.

    Args:
        composition (str/Composition): Either pymatgen Composition object or string formula,
            eg: "NaCl", "Na+1Cl-1", "Fe2+3O3-2" or "Fe2 +3 O3 -2"
            Notes:
                 - For 'ionic_radii' property, the Composition object must be made of oxidation
                    state decorated Specie objects not the plain Element objects.
                    eg.  fe2o3 = Composition({Specie("Fe", 3): 2, Specie("O", -2): 3})
                 - For string formula, the oxidation state sign(+ or -) must be specified explicitly.
                    eg.  "Fe2+3O3-2"

        property_name (str): pymatgen element attribute name, as defined in the Element class at
            http://pymatgen.org/_modules/pymatgen/core/periodic_table.html

    Returns:
        (list) of values containing descriptor floats for each atom in the compound(sorted by the
            electronegativity of the contituent atoms)

    """
    eldata = []
    # what are these named tuples for? not used or returned! -KM
    eldata_tup_lst = []
    eldata_tup = collections.namedtuple('eldata_tup', 'element propname propvalue propunit amt')

    oxidation_states = {}
    if isinstance(composition, Composition):
        # check whether the composition is composed of oxidation state decorates species (not just plain Elements)
        if hasattr(composition.elements[0], "oxi_state"):
            oxidation_states = dict([(str(sp.element), sp.oxi_state) for sp in composition.elements])
        el_amt_dict = composition.get_el_amt_dict()
    # string
    else:
        comp, oxidation_states = get_composition_oxidation_state(composition)
        el_amt_dict = comp.get_el_amt_dict()

    symbols = sorted(el_amt_dict.keys(), key=lambda sym: get_el_sp(sym).X)

    for el_sym in symbols:

        element = Element(el_sym)
        property_value = None
        property_units = None

        try:
            p = getattr(element, property_name)
        except AttributeError:
            print("{} attribute missing".format(property_name))
            raise

        if p is not None:
            if property_name in ['ionic_radii']:
                if oxidation_states:
                    property_value = element.ionic_radii[oxidation_states[el_sym]]
                    property_units = Unit("ang")
                else:
                    raise ValueError("oxidation state not given for {}; It does not yield a unique "
                                     "number per Element".format(property_name))
            else:
                property_value = float(p)

            # units are None for these pymatgen descriptors
            # todo: there seem to be a lot more unitless descriptors which are not listed here... -Alex D
            if property_name not in ['X', 'Z', 'group', 'row', 'number', 'mendeleev_no', 'ionic_radii']:
                property_units = p.unit

        # Make a named tuple out of all the available information
        eldata_tup_lst.append(eldata_tup(element=el_sym, propname=property_name, propvalue=property_value,
                                         propunit=property_units, amt=el_amt_dict[el_sym]))

        # Add descriptor values, one for each atom in the compound
        for i in range(int(el_amt_dict[el_sym])):
            eldata.append(property_value)

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
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                            'magpie_elementdata')
    magpiedata = []
    magpiedata_tup_lst = []
    magpiedata_tup = collections.namedtuple('magpiedata_tup',
                                            'element propname propvalue propunit amt')
    available_props = []

    # Make a list of available properties

    for datafile in os.listdir(data_dir):
        available_props.append(datafile.replace('.table', ''))

    if descriptor_name not in available_props:
        raise ValueError(
            "This descriptor is not available from the Magpie repository. Choose from {}".format(
                available_props))

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
            magpiedata_tup_lst.append(magpiedata_tup(element=el, propname=descriptor_name,
                                                     propvalue=float(lines[atomic_no - 1]),
                                                     propunit=unit,
                                                     amt=el_amt[el]))

            # Add descriptor values, one for each atom in the compound
            for i in range(int(el_amt[el])):
                magpiedata.append(float(lines[atomic_no - 1]))

    return magpiedata


def get_cohesive_energy(comp):
    """
    Get cohesive energy of compound by subtracting elemental cohesive energies from the formation energy of the compund.
    Elemental cohesive energies are taken from http://www.knowledgedoor.com/2/elements_handbook/cohesive_energy.html.
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


def get_composition_oxidation_state(formula):
    """
    Returns the composition and oxidation states from the given formula.
    Formula examples: "NaCl", "Na+1Cl-1",   "Fe2+3O3-2" or "Fe2 +3 O3 -2"

    Args:
        formula (str):

    Returns:
        pymatgen.core.composition.Composition, dict of oxidation states as strings

    """
    oxidation_states_dict = {}
    non_alphabets = re.split('[a-z]+', formula, flags=re.IGNORECASE)
    if not non_alphabets:
        return Composition(formula), oxidation_states_dict
    oxidation_states = []
    for na in non_alphabets:
        s = na.strip()
        if s != "" and ("+" in s or "-" in s):
            digits = re.split('[+-]+', s)
            sign_tmp = re.split('\d+', s)
            sign = [x.strip() for x in sign_tmp if x.strip() != ""]
            oxidation_states.append("{}{}".format(sign[-1], digits[-1].strip()))
    if not oxidation_states:
        return Composition(formula), oxidation_states_dict
    formula_plain = []
    before, after = tuple(formula.split(oxidation_states[0], 1))
    formula_plain.append(before)
    for oxs in oxidation_states[1:]:
        before, after = tuple(after.split(oxs, 1))
        formula_plain.append(before)
    for i, g in enumerate(formula_plain):
        el = re.split("\d", g.strip())[0]
        oxidation_states_dict[str(Element(el))] = int(oxidation_states[i])
    return Composition("".join(formula_plain)), oxidation_states_dict


if __name__ == '__main__':

    import pandas as pd

    descriptors = ['atomic_mass', 'X', 'Z', 'thermal_conductivity', 'melting_point',
                   'coefficient_of_linear_thermal_expansion']

    for desc in descriptors:
        print(get_pymatgen_descriptor('LiFePO4', desc))
    print(get_magpie_descriptor('LiFePO4', 'AtomicVolume'))
    print(get_magpie_descriptor('LiFePO4', 'Density'))
    print(get_holder_mean([1, 2, 3, 4], 0))

    training_set = pd.DataFrame({"composition": ["Fe2O3"]})
    print("WARD NPJ ATTRIBUTES")
    print("Stoichiometric attributes")
    p_list = [0, 2, 3, 5, 7, 9]
    print(StoichAttributes().featurize_all(training_set))
    print("Elemental property attributes")
    print(ElemPropertyAttributes().featurize_all(training_set))
    print("Valence Orbital Attributes")
    print(ValenceOrbitalAttributes().featurize_all(training_set))
    print("Ionic attributes")
    print(IonicAttributes().featurize_all(training_set))
