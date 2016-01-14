from pymatgen import Element, Composition

__author__ = 'Anubhav Jain <ajain@lbl.gov>'


def get_mass_list(x):
    el_mass = []
    el_amt = Composition(x).get_el_amt_dict()
    for el in el_amt:
        el_mass.append(Element(el).atomic_mass)
    return el_mass


if __name__ == '__main__':
    print get_mass_list('PbTe')
