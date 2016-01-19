from pymatgen import Element, Composition
import numpy as np

__author__ = 'Anubhav Jain <ajain@lbl.gov>'


def get_mass_list(comp):
    elmass = []
    el_amt = Composition(comp).get_el_amt_dict()
    print el_amt
    for el in el_amt:
        elmass.append(Element(el).atomic_mass)
    return elmass


def get_atomic_numbers(comp):
    elatomno = []
    el_amt = Composition(comp).get_el_amt_dict()
    for el in el_amt:
        elatomno.append(Element(el).Z)
    return elatomno


def get_pauling_elect(comp):
    electroneg = []
    el_amt = Composition(comp).get_el_amt_dict()
    for el in el_amt:
        electroneg.append(Element(el).X)
    return electroneg


def get_melting_pt(comp):
    melt_pts = []
    el_amt = Composition(comp).get_el_amt_dict()
    for el in el_amt:
        melt_pts.append(Element(el).melting_point)
    return melt_pts


def get_atomic_fraction(comp):
    el_amt = Composition(comp).get_el_amt_dict()
    for el in el_amt:
        el_amt[el] = Composition(comp).get_atomic_fraction(el)
    return el_amt


def get_wt_fraction(comp):
    el_amt = Composition(comp).get_el_amt_dict()
    for el in el_amt:
        el_amt[el] = Composition(comp).get_wt_fraction(el)
    return el_amt


def get_molar_volume(comp):
    molar_volumes = []
    el_amt = Composition(comp).get_el_amt_dict()
    for el in el_amt:
        str_vol = Element(el).molar_volume
        volume = float(str_vol.split()[0])
        molar_volumes.append(volume)
    return molar_volumes


def get_atomic_radius(comp):
    atomic_radii = []
    el_amt = Composition(comp).get_el_amt_dict()
    for el in el_amt:
        atomic_radii.append(Element(el).atomic_radius)
    return atomic_radii


def get_vanderwaals_radius(comp):
    vanderwaals_radii = []
    el_amt = Composition(comp).get_el_amt_dict()
    for el in el_amt:
        vanderwaals_radii.append(Element(el).van_der_waals_radius)
    return vanderwaals_radii


def get_averageionic_radius(comp):
    avgionic_radii = []
    el_amt = Composition(comp).get_el_amt_dict()
    for el in el_amt:
        avgionic_radii.append(Element(el).average_ionic_radius)
    return avgionic_radii


def get_ionic_radius(comp):
    ionic_radii = []
    el_amt = Composition(comp).get_el_amt_dict()
    for el in el_amt:
        ionic_radii.append(Element(el).ionic_radii)
    return ionic_radii


def get_max_min(lst):
    maxmin_dict = {'Max': max(lst), 'Min': min(lst)}
    return maxmin_dict


def get_mean(lst):
    return np.mean(lst)


def get_std(lst):
    return np.std(lst)


def get_med(lst):
    return np.median(lst)


def get_total(lst):
    return sum(lst)

if __name__ == '__main__':
    m = get_mass_list('LiFePO4')
    print m
    m_n = get_max_min(m)
    print m_n
    m_m = get_mean(m)
    print m_m
    m_s = get_std(m)
    print m_s
    m_md = get_med(m)
    print m_md
    m_a = get_atomic_numbers('LiFePO4')
    print m_a
    m_e = get_pauling_elect('LiFePO4')
    print m_e
    m_me = get_melting_pt('LiFePO4')
    print m_me
    m_af = get_atomic_fraction('LiFePO4')
    print m_af
    m_wf = get_wt_fraction('LiFePO4')
    print m_wf
    m_s = get_total(m)
    print m_s
    m_v = get_molar_volume('LiFePO4')
    print m_v
    m_ar = get_atomic_radius('LiFePO4')
    print m_ar
    m_vr = get_vanderwaals_radius('LiFePO4')
    print m_vr
    m_air = get_averageionic_radius('LiFePO4')
    print m_air
    m_ir = get_ionic_radius('LiFePO4')
    print m_ir


