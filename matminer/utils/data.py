from __future__ import division, unicode_literals, print_function

"""
Utility classes for retrieving elemental properties. Provides
a uniform interface to several different elemental property resources
including ``pymatgen`` and ``Magpie``.
"""

import os
import json
import six
import abc
import itertools
import numpy as np
from glob import glob
from collections import defaultdict

from monty.design_patterns import singleton

from pymatgen import Element, Composition
from pymatgen.core.periodic_table import _pt_data, get_el_sp

__author__ = 'Kiran Mathew, Jiming Chen, Logan Ward, Anubhav Jain'

module_dir = os.path.dirname(os.path.abspath(__file__))


class AbstractData(six.with_metaclass(abc.ABCMeta)):
    """Abstract class for retrieving elemental properties

    All classes must implement the `get_elemental_property` operation"""

    @abc.abstractmethod
    def get_elemental_property(self, elem, property_name):
        """Get a certain elemental property for a certain element.

        Args:
            elem - (Element) element to be assessed
            property_name - (str) property to be retreived
        Returns:
            float, property of that element
        """
        pass

    def get_elemental_properties(self, elems, property_name):
        """Get elemental properties for a list of elements

        Args:
            elems - ([Element]) list of elements
            property_name - (str) property to be retrieved
        Returns:
            [float], properties of elements
        """
        return [self.get_elemental_property(e, property_name) for e in elems]


class OxidationStatesMixin(six.with_metaclass(abc.ABCMeta)):
    """Abstract class interface for retrieving the oxidation states
    of each element"""

    @abc.abstractmethod
    def get_oxidation_states(self, elem):
        """Retrive the oxidation states of an element

        Args:
            elem - (Element), Target element
        Returns:
            [int] - oxidation states
        """
        pass


@singleton
class CohesiveEnergyData(AbstractData):
    """Get the cohesive energy of an element.

    TODO: Where is this data from? -wardlt
    """

    def __init__(self):
        # Load elemental cohesive energy data from json file
        with open(os.path.join(module_dir, 'data_files',
                               'cohesive_energies.json'), 'r') as f:
            self.cohesive_energy_data = json.load(f)

    def get_elemental_property(self, elem, property_name='cohesive energy'):
        """
        Args:
            x: (str) Element as str
            property_name (str): unused, always returns cohesive energy

        Returns:
            (float): cohesive energy of the element
        """
        return self.cohesive_energy_data[elem]


@singleton
class DemlData(AbstractData, OxidationStatesMixin):
    """
    Singleton class to get data from Deml data file. See also: A.M. Deml,
    R. O'Hayre, C. Wolverton, V. Stevanovic, Predicting density functional
    theory total energies and enthalpies of formation of metal-nonmetal
    compounds by linear regression, Phys. Rev. B - Condens. Matter Mater. Phys.
    93 (2016).
    """

    def __init__(self):
        from matminer.utils.data_files.deml_elementdata import properties
        self.all_props = properties
        self.available_props = list(self.all_props.keys()) + \
                               ["formal_charge", "valence_s", "valence_p",
                                "valence_d", "first_ioniz", "total_ioniz"]

        self.charge_dependent_properties = ["xtal_field_split", "magn_moment", "so_coupling", "first_ioniz"]

    def get_elemental_property(self, elem, property_name):
        if "valence" in property_name:
            valence_dict = self.all_props["valence_e"][
                self.all_props["col_num"][elem.symbol]]
            if property_name[-1] in ["s", "p", "d"]:
                # Return one of the shells
                return valence_dict[property_name[-1]]
            else:
                return sum(valence_dict.values())
        else:
            return self.all_props[property_name].get(elem.symbol, float("NaN"))

    def get_oxidation_states(self, elem):
        return self.all_props["charge_states"][elem.symbol]

    def calc_formal_charge(self, comp):
        """
        Computes formal charge of each element in a composition
        Args:
            comp (str or Composition object): composition
        Returns:
            dictionary of elements and formal charges
        """

        if type(comp) == str:
            comp = Composition(comp)

        el_amt = comp.get_el_amt_dict()
        symbols = sorted(el_amt.keys(), key=lambda sym: get_el_sp(
            sym).Z)  # Sort by atomic number
        stoich = [el_amt[el] for el in symbols]

        charge_states = []
        for el in symbols:
            try:
                charge_states.append(self.all_props["charge_states"][el])
            except:
                charge_states.append([float("NaN")])

        charge_sets = itertools.product(*charge_states)
        possible_fml_charge = []

        for charge in charge_sets:
            if np.dot(charge, stoich) == 0:
                possible_fml_charge.append(charge)

        if len(possible_fml_charge) == 0:
            fml_charge_dict = dict(zip(symbols, len(symbols) * [float("NaN")]))
        elif len(possible_fml_charge) == 1:
            fml_charge_dict = dict(zip(symbols, possible_fml_charge[0]))
        else:
            scores = []  # Score for correct sorting
            for charge_state in possible_fml_charge:
                el_charge_sort = [sym for (charge, sym) in sorted(
                    zip(charge_state, symbols))]  # Elements sorted by charge
                scores.append(sum(el_charge_sort[i] == symbols[i]) for i in
                              range(len(
                                  symbols)))  # Score based on number of elements in correct position

            fml_charge_dict = dict(zip(symbols, possible_fml_charge[
                scores.index(
                    max(scores))]))  # Get charge states with best score

        return fml_charge_dict


@singleton
class MagpieData(AbstractData, OxidationStatesMixin):
    """
    Singleton class to get data from Magpie files. See also:
    L. Ward, A. Agrawal, A. Choudhary, C. Wolverton, A general-purpose machine
    learning framework for predicting properties of inorganic materials,
    Npj Comput. Mater. 2 (2016) 16028.
    """

    def __init__(self):
        self.all_elemental_props = defaultdict(dict)
        self.available_props = []
        self.data_dir = os.path.join(module_dir, "data_files",
                                     'magpie_elementdata')

        # Make a list of available properties
        for datafile in glob(os.path.join(self.data_dir, "*.table")):
            self.available_props.append(
                os.path.basename(datafile).replace('.table', ''))

        # parse and store elemental properties
        for descriptor_name in self.available_props:
            with open(os.path.join(self.data_dir,
                                   '{}.table'.format(descriptor_name)),
                      'r') as f:
                lines = f.readlines()
                for atomic_no in range(1, len(_pt_data) + 1):  # max Z=103
                    try:
                        if descriptor_name in ["OxidationStates"]:
                            prop_value = [float(i) for i in
                                          lines[atomic_no - 1].split()]
                        else:
                            prop_value = float(lines[atomic_no - 1])
                    except ValueError:
                        prop_value = float("NaN")
                    self.all_elemental_props[descriptor_name][
                        Element.from_Z(atomic_no).symbol] = prop_value

    def get_elemental_property(self, elem, property_name):
        return self.all_elemental_props[property_name][elem.symbol]

    def get_oxidation_states(self, elem):
        return self.all_elemental_props["OxidationStates"][elem.symbol]


class PymatgenData(AbstractData, OxidationStatesMixin):
    """
    Class to get data from pymatgen. See also:
    S.P. Ong, W.D. Richards, A. Jain, G. Hautier, M. Kocher, S. Cholia, et al.,
    Python Materials Genomics (pymatgen): A robust, open-source python library
    for materials analysis, Comput. Mater. Sci. 68 (2013) 314-319.
    """

    def get_elemental_property(self, elem, property_name):
        if property_name == "block":
            block_key = {"s": 1.0, "p": 2.0, "d": 3.0, "f": 3.0}
            return block_key[getattr(elem, property_name)]
        else:
            return getattr(elem, property_name)

    def get_oxidation_states(self, elem: Element, common=True):
        """Get the oxidation states of an element

        Args:
            elem - (Element) target element
            common - (boolean), whether to return only the common oxidation states,
                or all known oxidation states
        Returns:
            [int] list of oxidation states
            """
        return elem.common_oxidation_states if common else elem.oxidation_states