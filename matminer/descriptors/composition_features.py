import collections
import json
import itertools

import numpy as np

import os
import re
from functools import reduce

from pymatgen import Element, Composition, MPRester
from pymatgen.core.units import Unit
from pymatgen.core.periodic_table import get_el_sp

__author__ = 'Saurabh Bajaj <sbajaj@lbl.gov>'

# TODO: read Magpie file only once
# TODO: unit tests
# TODO: most of this code needs to be rewritten ... AJ


# Load elemental cohesive energy data from json file
with open(os.path.join(os.path.dirname(__file__), 'cohesive_energies.json'), 'r') as f:
    ce_data = json.load(f)

#empty dictionary for magpie properties
magpie_props = {}

#list of elements
atomic_syms = []
for atomic_no in range(1,104):
    atomic_syms.append(Element.from_Z(atomic_no).symbol)

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
            if prop in ['X', 'Z', 'ionic_radii', 'group', 'row', 'number', 'mendeleev_no','oxidation_states','common_oxidation_states']:
                units = None
            else:
                units = getattr(Element(el), prop).unit

            # Make a named tuple out of all the available information
            eldata_tup_lst.append(
                eldata_tup(element=el, propname=prop, propvalue=float(getattr(Element(el), prop)), propunit=units,
                           amt=el_amt_dict[el]))

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

def get_magpie_dict(comp, descriptor_name):
    """
    Gets magpie data for an element. 
    First checks if magpie properties are already loaded, if not, stores magpie data in a dictionary

    Args: 
        elem (string): Atomic symbol of element
        descriptor_name (string): Name of descriptor

    Returns:
        attr_dict
    """

    if descriptor_name not in magpie_props:

        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", 'magpie_elementdata')
        available_props = []

        # Make a list of available properties
        for datafile in os.listdir(data_dir):
            available_props.append(datafile.replace('.table', ''))

        if descriptor_name not in available_props:
            raise ValueError(
                "This descriptor is not available from the Magpie repository. Choose from {}".format(available_props))

    #if descriptor_name not in magpie_props:
        # Extract from data file
        prop_value = []
        with open(os.path.join(data_dir, '{}.table'.format(descriptor_name)), 'r') as descp_file:
            lines = descp_file.readlines()
            
            for atomic_no in range(1, 104): #This is as high as pymatgen goes
                try:
                    if descriptor_name in ["OxidationStates"]:
                        prop_value.append([float(i) for i in lines[atomic_no - 1].split()])
                    else: 
                        prop_value.append(float(lines[atomic_no - 1]))
                except:
                    prop_value.append(float("NaN"))
        
        attr_dict = dict(zip(atomic_syms, prop_value))    
        magpie_props[descriptor_name] = attr_dict #Add dictionary to magpie_props

    ##Get data for given element/compound
    el_amt = Composition(comp).get_el_amt_dict()
    elements = list(el_amt.keys())
    
    magpiedata = []
    for el in elements:
        for i in range(int(el_amt[el])):
            magpiedata.append(magpie_props[descriptor_name][el])
 
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

###Stoichiometric attributes from Ward npj paper
def get_stoich_attributes(comp, p):
    """
    Get stoichiometric attributes

    Args:
        comp (string): Chemical formula
        p (int)
    
    Returns: 
        p_norm (float): Lp norm-based stoichiometric attribute
    """

    el_amt = Composition(comp).get_el_amt_dict()
    
    p_norm = 0
    n_atoms = sum(el_amt.values())

    if p < 0:
        raise ValueError("p-norm not defined for p < 0")

    if p == 0:
        p_norm = len(el_amt.values())
    else:
        for i in el_amt:
            p_norm += (el_amt[i]/n_atoms)**p
        p_norm = p_norm**(1.0/p)

    return p_norm

###Elemental properties from Ward npj paper
def get_elem_property_attributes(comp):
    """
    Get elemental property attributes

    Args:

        comp (string): Chemical formula
    
    Returns:
        all_attributes: min, max, range, mean, average deviation, and mode of 22 descriptors
    """

    comp_obj = Composition(comp)
    el_amt = comp_obj.get_el_amt_dict()
    elements = list(el_amt.keys())

    #atom_frac = []
    #for elem in elements:
        #atom_frac.append(comp_obj.get_atomic_fraction(elem))

    req_data = ["Number", "MendeleevNumber", "AtomicWeight","MeltingT","Column","Row","CovalentRadius","Electronegativity",
        "NsValence","NpValence","NdValence","NfValence","NValance","NsUnfilled","NpUnfilled","NdUnfilled","NfUnfilled","NUnfilled",
        "GSvolume_pa","GSbandgap","GSmagmom","SpaceGroupNumber"]    
    
    all_attributes = []

    for attr in req_data:
        
        elem_data = get_magpie_dict(comp, attr)

        #for elem in elements:
        #    elem_data.append(get_magpie_dict(elem, attr)[0])

        desc_stats = []
        desc_stats.append(min(elem_data))
        desc_stats.append(max(elem_data))
        desc_stats.append(max(elem_data) - min(elem_data))
        
        prop_mean = np.mean(elem_data)
        desc_stats.append(prop_mean)
        desc_stats.append(np.mean(np.abs(np.subtract(elem_data, prop_mean))))
        desc_stats.append(max(set(elem_data), key=elem_data.count))
        all_attributes.append(desc_stats)

    return all_attributes

def get_valence_orbital_attributes(comp):
    """Weighted fraction of valence electrons in each orbital

       Args: 
            comp (string): Chemical formula

       Returns: 
            Fs, Fp, Fd, Ff (float): Fraction of valence electrons in s, p, d, and f orbitals
    """    

    avg_total_valence = np.mean(get_magpie_dict(comp, "NValance"))
    avg_s = np.mean(get_magpie_dict(comp, "NsValence"))
    avg_p = np.mean(get_magpie_dict(comp, "NpValence"))
    avg_d = np.mean(get_magpie_dict(comp, "NdValence"))
    avg_f = np.mean(get_magpie_dict(comp, "NfValence"))

    Fs = avg_s/avg_total_valence
    Fp = avg_p/avg_total_valence
    Fd = avg_d/avg_total_valence
    Ff = avg_f/avg_total_valence

    return Fs, Fp, Fd, Ff

def get_ionic_attributes(comp):
    """
    Ionic character attributes

    Args:
        comp (string): Chemical formula

    Returns:
        cpd_possible (bool): Indicates if a neutral ionic compound is possible
        max_ionic_char (float): Maximum ionic character between two atoms
        avg_ionic_char (float): Average ionic character
    """
    comp_obj = Composition(comp)
    el_amt = comp_obj.get_el_amt_dict()
    elements = list(el_amt.keys())
    values = list(el_amt.values())
 
    #Determine if it is possible to form neutral ionic compound
    ox_states = []
    for elem in elements:
        ox_states.append(get_magpie_dict(elem,"OxidationStates")[0])
    
    cpd_possible = False
    ox_sets = itertools.product(*ox_states)
    for ox in ox_sets:
        if np.dot(ox, values) == 0:
            cpd_possible = True
            break    

    atom_pairs = itertools.combinations(elements, 2)

    ionic_char = []
    avg_ionic_char = 0

    for pair in atom_pairs:
        XA = get_magpie_dict(pair[0], "Electronegativity")
        XB = get_magpie_dict(pair[1], "Electronegativity")
        ionic_char.append(1.0 - np.exp(-0.25*(XA[0]-XB[0])**2))
        avg_ionic_char += comp_obj.get_atomic_fraction(pair[0])*comp_obj.get_atomic_fraction(pair[1])*ionic_char[-1]
    
    max_ionic_char = np.max(ionic_char)
 
    return cpd_possible, max_ionic_char, avg_ionic_char

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
    descriptors = ['atomic_mass', 'X', 'Z', 'thermal_conductivity', 'melting_point',
                   'coefficient_of_linear_thermal_expansion']

    for desc in descriptors:
        print(get_pymatgen_descriptor('LiFePO4', desc))
    print(get_magpie_descriptor('LiFePO4', 'AtomicVolume'))
    print(get_magpie_descriptor('LiFePO4', 'Density'))
    print(get_holder_mean([1, 2, 3, 4], 0))
    
    ####TESTING WARD NPJ DESCRIPTORS
    print "WARD NPJ ATTRIBUTES"
    print "Stoichiometric attributes"
    print get_stoich_attributes("Fe2O3", 3)
    print "Elemental property attributes"
    print get_elem_property_attributes("Fe2O3")
    print "Valence Orbital Attributes"
    print get_valence_orbital_attributes("Fe2O3")
    print "Ionic attributes"
    print get_ionic_attributes("Fe2O3")
