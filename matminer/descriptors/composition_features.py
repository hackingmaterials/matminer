from pymatgen import Element, Composition, FloatWithUnit
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

# TODO: Handle numbers with units, eg. thermal_conductivity
# TODO: Handle dictionaries in case of atomic radii
# TODO: Handle None values

def get_pymatgen_eldata_lst(comp, prop):
    eldata_lst = []
    eldata = collections.namedtuple('eldata', 'element propname propvalue amt')
    el_amt_dict = Composition(comp).get_el_amt_dict()
    for el in el_amt_dict:
        if callable(getattr(Element(el), prop)) is None:
            print 'Invalid pymatgen Element attribute(property)'
            return
        eldata_lst.append(eldata(element=el, propname=prop, propvalue=getattr(Element(el), prop), amt=el_amt_dict[el]))
    return eldata_lst


def get_thermal_cond(comp):
    thermalcond = []
    el_amt = Composition(comp).get_el_amt_dict()
    for el in el_amt:
        print FloatWithUnit.from_string(Element(el).thermal_conductivity)
        # tc = Element(el).thermal_conductivity
        # thermalcond.append(float(tc.split()[0]))
    return thermalcond


def get_melting_pt(comp):
    melt_pts = []
    el_amt = Composition(comp).get_el_amt_dict()
    for el in el_amt:
        melt_pts.append(float(re.findall('[-+]?\d+[\.]?\d*', Element(el).melting_point)[0]) * el_amt[el])
    return melt_pts


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


def get_linear_thermal_expansion(comp):
    thermal_exp = []
    thermaldata = collections.namedtuple('thermaldata', 'element propname propvalue amt')
    el_amt_dict = Composition(comp).get_el_amt_dict()
    for el in el_amt_dict:
        exp = Element(el).coefficient_of_linear_thermal_expansion
        if exp is not None:
            # thermal_exp.append([float(exp.split()[0]), el_amt_dict[el]])
            thermal_exp.append(thermaldata(element=el, propname='coefficient_of_linear_thermal_expansion',
                                           propvalue=float(exp.split()[0]), amt=el_amt_dict[el]))
    return thermal_exp


def get_max_min(lst):
    maxmin = {'Max': max(lst), 'Min': min(lst)}
    return maxmin


def get_mean(lst):
    total_propamt = 0
    total_amt = 0
    for element in lst:
        total_propamt += (element.propvalue * element.amt)
        total_amt += element.amt
    return total_propamt / total_amt


def get_std(lst):
    mean = get_mean(lst)
    total_weighted_squares = 0
    total_amt = 0
    for element in lst:
        total_weighted_squares += (element.amt * (element.propvalue - mean) ** 2)
        total_amt += element.amt
    return math.sqrt(total_weighted_squares / total_amt)


def get_med(lst):
    return np.median(lst)


def get_total(lst):
    return sum(lst)


if __name__ == '__main__':
    descriptors = ['ionic_radii', 'atomic_mass', 'X', 'Z']
    for desc in descriptors:
        print get_pymatgen_eldata_lst('LiFePO4', desc)
    print get_std(get_linear_thermal_expansion('LiFePO4'))
