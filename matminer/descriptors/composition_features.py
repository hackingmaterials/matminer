from pymatgen import Element, Composition

__author__ = 'Anubhav Jain <ajain@lbl.gov>'


def get_mass_list(comp):
    el_mass = []
    el_amt = Composition(comp).get_el_amt_dict()
    for el in el_amt:
        el_mass.append(Element(el).atomic_mass)
    return el_mass

def get_max_min(lst):
    maxmin_dict = {}
    maxmin_dict['Max'] = max(lst)
    maxmin_dict['Min'] = min(lst)
    return maxmin_dict

if __name__ == '__main__':
    m = get_mass_list('LiFePO4')
    print m
    n = get_max_min(m)
    print n
