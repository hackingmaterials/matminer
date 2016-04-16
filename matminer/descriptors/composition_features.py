from pymatgen import Element, Composition
import numpy as np
import re
import math
import collections
import inspect

__author__ = 'Saurabh Bajaj <sbajaj@lbl.gov>'


# AJ says: I am 90% sure that 90% of these methods are completely pointless if you just support pymatgen w/units
# inside Pandas dataframes
# AJ says: Plus, 90% of these methods are completely wrong. e.g. elmass.append(Element(el).atomic_mass * el_amt[el])
# is WRONG

# TODO: merge almost all methods to 1
# TODO: read Magpie file only once
# TODO: NamedTuple or tuple
# TODO: unit tests
# TODO: use pymatgen Units class to handle units well (don't parse strings manually)

# TODO: Handle dictionaries in case of atomic radii
# TODO: Handle None values

def get_pymatgen_eldata_lst(comp, prop):
    eldata_lst = []
    eldata = collections.namedtuple('eldata', 'element propname propvalue propunit amt')
    el_amt_dict = Composition(comp).get_el_amt_dict()
    for el in el_amt_dict:
        if callable(getattr(Element(el), prop)) is None:
            print 'Invalid pymatgen Element attribute(property)'
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
    descriptor_list = []
    el_amt = Composition(comp).get_el_amt_dict()
    descp_file = open('data/magpie_elementdata/' + descriptor_name + '.table', 'r')
    lines = descp_file.readlines()
    for el in el_amt:
        atomic_no = Element(el).Z
        descriptor_list.append(float(lines[atomic_no - 1]))
    descp_file.close()
    return descriptor_list


def get_stats(lst):
    propvalues = []
    for element in lst:
        if element.propvalue is not None:
            propvalues.append(element.propvalue)
    return {'max': max(propvalues), 'min': min(propvalues), 'median': np.median(propvalues), 'sum': sum(propvalues)}


def get_mean(lst):
    total_propamt = 0
    total_amt = 0
    for element in lst:
        if element.propvalue is not None:
            total_propamt += (element.propvalue * element.amt)
            total_amt += element.amt
    return total_propamt / total_amt


def get_std(lst):
    mean = get_mean(lst)
    total_weighted_squares = 0
    total_amt = 0
    for element in lst:
        if element.propvalue is not None:
            total_weighted_squares += (element.amt * (element.propvalue - mean) ** 2)
            total_amt += element.amt
    return math.sqrt(total_weighted_squares / total_amt)


if __name__ == '__main__':
    descriptors = ['atomic_mass', 'X', 'Z', 'thermal_conductivity', 'melting_point',
                   'coefficient_of_linear_thermal_expansion']
    # 'ionic_radii',
    for desc in descriptors:
        print get_pymatgen_eldata_lst('LiFePO4', desc)
