from pymatgen import Element, Composition
import numpy as np
import math
import collections
import os

__author__ = 'Saurabh Bajaj <sbajaj@lbl.gov>'


# TODO: read Magpie file only once
# TODO: Handle dictionaries in case of atomic radii. Aj says "You can require that getting the ionic_radii descriptor
#  requires a valence-decorated Structure or valence-decorated Composition. Otherwise it does not work, i.e. returns
# None. Other radii (e.g. covalent) won't require an oxidation state and people can and should use those for
# non-ionic structures. You can also have a function that returns a mean of ionic_radii for all valences but that
# should not be the default."
# TODO: unit tests


def get_pymatgen_descriptor(comp, prop):
    """
    Get descriptor data for elements in a compound from pymatgen.

    :param comp: (str) compound composition, eg: "NaCl"
    :param prop: (str) pymatgen element attribute, as defined in the Element class at
    http://pymatgen.org/_modules/pymatgen/core/periodic_table.html
    :return: (list) of namedtuples containing element name, property name, property value, units, and amount of element
    """
    eldata_lst = []
    eldata = collections.namedtuple('eldata', 'element propname propvalue propunit amt')
    el_amt_dict = Composition(comp).get_el_amt_dict()
    for el in el_amt_dict:
        if callable(getattr(Element(el), prop)) is None:
            print('Invalid pymatgen Element attribute(property)')
            return
        if getattr(Element(el), prop) is not None:
            if prop in ['X', 'Z', 'ionic_radii']:
                units = None
            else:
                units = getattr(Element(el), prop).unit
            eldata_lst.append(
                eldata(element=el, propname=prop, propvalue=float(getattr(Element(el), prop)), propunit=units,
                       amt=el_amt_dict[el]))
        else:
            eldata_lst.append(eldata(element=el, propname=prop, propvalue=None, propunit=None, amt=el_amt_dict[el]))
    return eldata_lst


def get_magpie_descriptor(comp, descriptor_name):
    """
    Get descriptor data for elements in a compound from the Magpie data repository.

    :param comp: (str) compound composition, eg: "NaCl"
    :param descriptor_name: name of Magpie descriptor needed. Find the entire list at
    https://bitbucket.org/wolverton/magpie/src/6ecf8d3b79e03e06ef55c141c350a08fbc8da849/Lookup%20Data/?at=master
    :return: (list) of descriptor values for each element in the composition
    """
    available_props = []
    for datafile in os.listdir('data/magpie_elementdata'):
        available_props.append(datafile.replace('.table', ''))
    if descriptor_name not in available_props:
        print('This descriptor is not available from the Magpie repository. '
              'Choose from {}'.format(available_props))
        return
    descriptor_list = []
    el_amt = Composition(comp).get_el_amt_dict()
    with open('data/magpie_elementdata/' + descriptor_name + '.table', 'r') as descp_file:
        lines = descp_file.readlines()
        for el in el_amt:
            atomic_no = Element(el).Z
            descriptor_list.append(float(lines[atomic_no - 1]))
    return descriptor_list


def get_maxmin(lst):
    """
    Get maximum, minimum, median, and total of descriptor values.

    :param lst: (list) of namedtuples as output by get_element_data()
    :return: (dict) containing maximum, minimum, median, and total of all property values
    """
    propvalues = []
    for element in lst:
        if element.propvalue is not None:
            propvalues.append(element.propvalue)
    return {'max': max(propvalues), 'min': min(propvalues), 'median': np.median(propvalues), 'sum': sum(propvalues)}


def get_mean(lst):
    """
    Get mean of descriptor values.

    :param lst: (list) of namedtuples as output by get_element_data()
    :return: (float) weighted mean of property values
    """
    total_propamt = 0
    total_amt = 0
    for element in lst:
        if element.propvalue is not None:
            total_propamt += (element.propvalue * element.amt)
            total_amt += element.amt
    return total_propamt / total_amt


def get_std(lst):
    """
    Get standard deviation of descriptor values.

    :param lst: (list) of namedtuples as output by get_element_data()
    :return: (float) weighted standard deviation of property values
    """
    mean = get_mean(lst)
    total_weighted_squares = 0
    total_amt = 0
    for element in lst:
        if element.propvalue is not None:
            total_weighted_squares += (element.amt * (element.propvalue - mean) ** 2)
            total_amt += element.amt
    return math.sqrt(total_weighted_squares / total_amt)


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


if __name__ == '__main__':
    descriptors = ['atomic_mass', 'X', 'Z', 'thermal_conductivity', 'melting_point',
                   'coefficient_of_linear_thermal_expansion']

    for desc in descriptors:
        print(get_pymatgen_descriptor('LiFePO4', desc))
    print(get_magpie_descriptor('LiFePO4', 'Atomicolume'))
