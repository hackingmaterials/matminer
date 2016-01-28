from pymatgen import Element, Composition
import numpy as np
import re

__author__ = 'Anubhav Jain <ajain@lbl.gov>'


# TODO: what is the purpose of this function? Discuss w/AJ
def get_el_amt(comp):
    return Composition(comp).get_el_amt_dict()


def get_masses(comp):
    elmass = []
    el_amt = Composition(comp).get_el_amt_dict()
    for el in el_amt:
        elmass.append(Element(el).atomic_mass * el_amt[el])
    return elmass


def get_atomic_numbers(comp):
    elatomno = []
    el_amt = Composition(comp).get_el_amt_dict()
    for el in el_amt:
        elatomno.append(Element(el).Z * el_amt[el])
    return elatomno


def get_pauling_elect(comp):
    electroneg = []
    el_amt = Composition(comp).get_el_amt_dict()
    for el in el_amt:
        electroneg.append(Element(el).X * el_amt[el])
    return electroneg


def get_melting_pt(comp):
    melt_pts = []
    el_amt = Composition(comp).get_el_amt_dict()
    for el in el_amt:
        melt_pts.append(float(re.findall('[-+]?\d+[\.]?\d*', Element(el).melting_point)[0]) * el_amt[el])
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
        molar_volumes.append(volume * el_amt[el])
    return molar_volumes


def get_atomic_radius(comp):
    atomic_radii = []
    el_amt = Composition(comp).get_el_amt_dict()
    for el in el_amt:
        atomic_radii.append(Element(el).atomic_radius)
    return atomic_radii


def get_calc_atomic_radius(comp):
    calc_atomic_radii = []
    el_amt = Composition(comp).get_el_amt_dict()
    for el in el_amt:
        calc_atomic_radii.append(Element(el).atomic_radius_calculated)
    return calc_atomic_radii


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


def get_magpie_descriptor(comp, descriptor_name):
    descriptor_list = []
    el_amt = Composition(comp).get_el_amt_dict()
    descp_file = open('data/magpie_elementdata/' + descriptor_name + '.table', 'r')
    lines = descp_file.readlines()
    for el in el_amt:
        atomic_no = Element(el).Z
        descriptor_list.append(float(lines[atomic_no-1]))
    descp_file.close()
    return descriptor_list


def get_max_oxidation_state(comp):
    max_oxi = []
    el_amt = Composition(comp).get_el_amt_dict()
    for el in el_amt:
        max_oxi.append(Element(el).max_oxidation_state)
    return max_oxi


def get_min_oxidation_state(comp):
    min_oxi = []
    el_amt = Composition(comp).get_el_amt_dict()
    for el in el_amt:
        min_oxi.append(Element(el).min_oxidation_state)
    return min_oxi


def get_oxidation_state(comp):
    oxi_states = []
    el_amt = Composition(comp).get_el_amt_dict()
    for el in el_amt:
        oxi_states.append(Element(el).oxidation_states)
    return oxi_states


def get_common_oxidation_state(comp):
    oxi_states = []
    el_amt = Composition(comp).get_el_amt_dict()
    for el in el_amt:
        oxi_states.append(Element(el).common_oxidation_states)
    return oxi_states


def get_full_elect_struct(comp):
    elect_config = []
    el_amt = Composition(comp).get_el_amt_dict()
    for el in el_amt:
        elect_config.append(Element(el).full_electronic_structure)
    return elect_config


def get_row(comp):
    row = []
    el_amt = Composition(comp).get_el_amt_dict()
    for el in el_amt:
        row.append(Element(el).row)
    return row


def get_group(comp):
    group = []
    el_amt = Composition(comp).get_el_amt_dict()
    for el in el_amt:
        group.append(Element(el).group)
    return group


def get_block(comp):
    block = []
    el_amt = Composition(comp).get_el_amt_dict()
    for el in el_amt:
        block.append(Element(el).block)
    return block


def get_mendeleev_no(comp):
    mendeleev = []
    el_amt = Composition(comp).get_el_amt_dict()
    for el in el_amt:
        mendeleev.append(Element(el).mendeleev_no)
    return mendeleev


def get_elec_res(comp):
    elec_res = []
    el_amt = Composition(comp).get_el_amt_dict()
    for el in el_amt:
        elec_res.append(Element(el).electrical_resistivity)
    return elec_res


# TODO: Check if this needs to be converted to a number
def get_reflectivity(comp):
    reflectivity = []
    el_amt = Composition(comp).get_el_amt_dict()
    for el in el_amt:
        reflectivity.append(Element(el).reflectivity)
    return reflectivity


def get_refractive_idx(comp):
    refac_idx = []
    el_amt = Composition(comp).get_el_amt_dict()
    for el in el_amt:
        idx = Element(el).refractive_index
        if idx is not None:
            refac_idx.append(float(idx))
        else:
            refac_idx.append(idx)
    return refac_idx


def get_poissons_ratio(comp):
    p_ratio = []
    el_amt = Composition(comp).get_el_amt_dict()
    for el in el_amt:
        poissons = Element(el).poissons_ratio
        if poissons is not None:
            p_ratio.append(float(poissons))
        else:
            p_ratio.append(poissons)
    return p_ratio


def get_thermal_cond(comp):
    thermalcond = []
    el_amt = Composition(comp).get_el_amt_dict()
    for el in el_amt:
        tc = Element(el).thermal_conductivity
        thermalcond.append(float(tc.split()[0]))
    return thermalcond


def get_boiling_pt(comp):
    boiling_pts = []
    el_amt = Composition(comp).get_el_amt_dict()
    for el in el_amt:
        bp = Element(el).boiling_point
        if bp is not None:
            boiling_pts.append(float(bp.split()[0]))
        else:
            boiling_pts.append(bp)
    return boiling_pts


def get_critical_temp(comp):
    critical_t = []
    el_amt = Composition(comp).get_el_amt_dict()
    for el in el_amt:
        ct = Element(el).critical_temperature
        # critical_t.append(ct)
        if ct is not None:
            critical_t.append(float(ct.split()[0]))
        else:
            critical_t.append(ct)
    return critical_t


def get_supercond_temp(comp):
    supercond_t = []
    el_amt = Composition(comp).get_el_amt_dict()
    for el in el_amt:
        sct = Element(el).superconduction_temperature
        # supercond_t.append(sct)
        if sct is not None:
            supercond_t.append(float(sct.split()[0]))
        else:
            supercond_t.append(sct)
    return supercond_t


def get_max_min(lst):
    maxmin = {'Max': max(lst), 'Min': min(lst)}
    return maxmin


def get_mean(lst):
    return np.mean(lst)


def get_std(lst):
    return np.std(lst)


def get_med(lst):
    return np.median(lst)


def get_total(lst):
    return sum(lst)

if __name__ == '__main__':
    print get_el_amt('LiFePO4')
    print get_masses('LiFePO4')
    print get_atomic_numbers('LiFePO4')
    print get_pauling_elect('LiFePO4')
    print get_melting_pt('LiFePO4')
    print get_atomic_fraction('LiFePO4')
    print get_wt_fraction('LiFePO4')
    print get_molar_volume('LiFePO4')
    print get_atomic_radius('LiFePO4')
    print get_calc_atomic_radius('LiFePO4')
    print get_vanderwaals_radius('LiFePO4')
    print get_averageionic_radius('LiFePO4')
    print get_ionic_radius('LiFePO4')
    print get_magpie_descriptor('LiFePO4', 'AtomicVolume')
    print get_max_oxidation_state('LiFePO4')
    print get_min_oxidation_state('LiFePO4')
    print get_oxidation_state('LiFePO4')
    print get_common_oxidation_state('LiFePO4')
    print get_full_elect_struct('LiFePO4')
    print get_row('LiFePO4')
    print get_group('LiFePO4')
    print get_block('LiFePO4')
    print get_mendeleev_no('LiFePO4')
    print get_elec_res('LiFePO4')
    print get_reflectivity('LiFePO4')
    print get_refractive_idx('PbTe')
    print get_poissons_ratio('LiFePO4')
    print get_thermal_cond('LiFePO4')
    print get_boiling_pt('LiFePO4')
    print get_critical_temp('LiFePO4')
    print get_supercond_temp('MgB2')