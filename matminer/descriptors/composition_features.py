import collections
import json
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


def get_pymatgen_descriptor(formula, property_name):
    """
    Get descriptor data for elements in a compound from pymatgen.

    Args:
        formula (str): compound formula, eg: "NaCl", "Na+1Cl-1", "Fe2+3O3-2" or "Fe2 +3 O3 -2"
            Note: the oxidation state sign(+ or -) must be specified
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
    comp, oxidation_states = get_composition_oxidation_state(formula)
    el_amt_dict = Composition(comp).get_el_amt_dict()
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
    descriptors = ['atomic_mass', 'X', 'Z', 'thermal_conductivity', 'melting_point',
                   'coefficient_of_linear_thermal_expansion']

    for desc in descriptors:
        print(get_pymatgen_descriptor('LiFePO4', desc))
    print(get_magpie_descriptor('LiFePO4', 'AtomicVolume'))
    print(get_magpie_descriptor('LiFePO4', 'Density'))
    print(get_holder_mean([1, 2, 3, 4], 0))
