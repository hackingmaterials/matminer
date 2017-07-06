from __future__ import division, unicode_literals, print_function

import itertools

import numpy as np

from matminer.descriptors.base import AbstractFeaturizer
from matminer.descriptors.data import magpie_data
from matminer.descriptors.utils import get_holder_mean

__author__ = 'Jimin Chen, Logan Ward, Saurabh Bajaj, Anubhav jain, Kiran Mathew'


class StoichiometricAttribute(AbstractFeaturizer):
    """
    Class to calculate stoichiometric attributes.

    Generates: Lp norm-based stoichiometric attribute.

    Args:
        p_list (list of ints): list of norms to calculate
    """

    def __init__(self, p_list=None):
        if p_list is None:
            self.p_list = [0, 2, 3, 5, 7, 10]
        else:
            self.p_list = p_list

    def featurize(self, comp):
        el_amt = comp.get_el_amt_dict()

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


class ElementalAttribute(AbstractFeaturizer):
    """
    Class to calculate elemental property attributes.

    Generates: list representation with min, max, range, mean,  average deviation, and
        mode of descriptors

    Args:
        properties (list of strings): List of elemental properties to use
    """

    def __init__(self, properties=None):
        if properties is None:
            self.properties = ["Number", "MendeleevNumber", "AtomicWeight", "MeltingT", "Column",
                               "Row", "CovalentRadius", "Electronegativity",
                               "NsValence", "NpValence", "NdValence", "NfValence", "NValance",
                               "NsUnfilled", "NpUnfilled", "NdUnfilled", "NfUnfilled", "NUnfilled",
                               "GSvolume_pa", "GSbandgap", "GSmagmom", "SpaceGroupNumber"]
        else:
            self.properties = properties

    def featurize(self, comp):
        all_properties = []

        for attr in self.properties:
            elem_data = magpie_data.get_property(comp, attr)

            all_properties.append(min(elem_data))
            all_properties.append(max(elem_data))
            all_properties.append(max(elem_data) - min(elem_data))

            prop_mean = sum(elem_data) / len(elem_data)
            all_properties.append(prop_mean)
            all_properties.append(sum(np.abs(np.subtract(elem_data, prop_mean))) / len(elem_data))
            all_properties.append(max(set(elem_data), key=elem_data.count))

        return all_properties

    def generate_labels(self):
        labels = []
        for attr in self.properties:
            labels.append("Min %s" % attr)
            labels.append("Max %s" % attr)
            labels.append("Range %s" % attr)
            labels.append("Mean %s" % attr)
            labels.append("AbsDev %s" % attr)
            labels.append("Mode %s" % attr)
        return labels


class ValenceOrbitalAttribute(AbstractFeaturizer):
    """
    Class to calculate valence orbital attributes.
    Generate fraction of valence electrons in s, p, d, and f orbitals
    """

    def __init__(self):
        pass

    def featurize(self, comp):
        num_atoms = comp.num_atoms

        avg_total_valence = sum(magpie_data.get_property(comp, "NValance")) / num_atoms
        avg_s = sum(magpie_data.get_property(comp, "NsValence")) / num_atoms
        avg_p = sum(magpie_data.get_property(comp, "NpValence")) / num_atoms
        avg_d = sum(magpie_data.get_property(comp, "NdValence")) / num_atoms
        avg_f = sum(magpie_data.get_property(comp, "NfValence")) / num_atoms

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


class IonicAttribute(AbstractFeaturizer):
    """
    Class to calculate ionic property attributes.

    Generates: [ cpd_possible (boolean value indicating if a neutral ionic compound is possible),
                 max_ionic_char (float value indicating maximum ionic character between two atoms),
                 avg_ionic_char (Average ionic character ]
    """

    def __init__(self):
        pass

    def featurize(self, comp):
        el_amt = comp.get_el_amt_dict()
        elements = list(el_amt.keys())
        values = list(el_amt.values())

        if len(elements) < 2:  # Single element
            cpd_possible = True
            max_ionic_char = 0
            avg_ionic_char = 0
        else:
            # Get magpie data for each element
            all_ox_states = magpie_data.get_property(comp, "OxidationStates")
            all_elec = magpie_data.get_property(comp, "Electronegativity")
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


if __name__ == '__main__':

    import pandas as pd

    descriptors = ['atomic_mass', 'X', 'Z', 'thermal_conductivity', 'melting_point',
                   'coefficient_of_linear_thermal_expansion']

    for desc in descriptors:
        print(get_pymatgen_descriptor('LiFePO4', desc))
    print(magpie_data.get_property('LiFePO4', 'AtomicVolume'))
    print(magpie_data.get_property('LiFePO4', 'Density'))
    print(get_holder_mean([1, 2, 3, 4], 0))

    training_set = pd.DataFrame({"composition": ["Fe2O3"]})
    print("WARD NPJ ATTRIBUTES")
    print("Stoichiometric attributes")
    p_list = [0, 2, 3, 5, 7, 9]
    print(StoichiometricAttribute().featurize_all(training_set))
    print("Elemental property attributes")
    print(ElementalAttribute().featurize_all(training_set))
    print("Valence Orbital Attributes")
    print(ValenceOrbitalAttribute().featurize_all(training_set))
    print("Ionic attributes")
    print(IonicAttribute().featurize_all(training_set))
