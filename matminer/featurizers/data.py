from __future__ import division, unicode_literals, print_function

# TODO: this code is all quite messy ... needs some beautification for sure

"""
Defines wrappers for data sources(magpi, pymatgen etc) for elemental properties.
"""

import os
import json
import re
import six
import abc
import itertools
import numpy as np
from glob import glob
from collections import defaultdict, namedtuple

from monty.design_patterns import singleton

from pymatgen import Element, Composition, Unit
from pymatgen.core.periodic_table import _pt_data, get_el_sp

__author__ = 'Kiran Mathew, Jiming Chen, Logan Ward'

module_dir = os.path.dirname(os.path.abspath(__file__))


class AbstractData((six.with_metaclass(abc.ABCMeta))):
    @abc.abstractmethod
    def get_property(self, x, property_name, combine_by_element=False):
        """
        Gets data for a composition object.

        Args:
            x: input data/object to retrieve the property of, e.g., an Element,
                Composition, or str representation
            property_name (str): Name of property to retrieve
            combine_by_element (bool): If true, behavior will be to collapse
                all values for a single Element to one value (e.g., VO and VO2
                will give the same data vector)

        Returns:
            Property value(s), typically either float or list of float.
            Note: if multiple values are returned for a Composition, the
            convention is to sort values by the element's atomic number.
        """
        pass


@singleton
class CohesiveEnergyData(AbstractData):
    """
    Singleton class to get cohesive energy data
    """

    def __init__(self):
        # Load elemental cohesive energy data from json file
        with open(os.path.join(module_dir, 'data_files',
                               'cohesive_energies.json'), 'r') as f:
            self.cohesive_energy_data = json.load(f)

    def get_property(self, x, property_name="cohesive_energy",
                     combine_by_element=True):
        """
        Args:
            x: (str) Element as str
            property_name (str): must be "cohesive_energy"
            combine_by_element (bool): must be True

        Returns:
            (float): cohesive energy of the element
        """
        return self.cohesive_energy_data[x]


@singleton
class DemlData(AbstractData):
    """
    Singleton class to get data from Deml data file
    """

    def __init__(self):
        from matminer.featurizers.data_files.deml_elementdata import properties
        self.all_props = properties
        self.available_props = list(self.all_props.keys()) + \
                               ["formal_charge", "valence_s", "valence_p",
                                "valence_d", "first_ioniz", "total_ioniz"]

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

    def get_property(self, comp, property_name, combine_by_element=False):
        """
        Args:
            x: (comp) Composition object (or str representation)
            property_name (str):
            combine_by_element (bool):

        Returns:
            (list): list of property values for the composition
        """
        if property_name not in self.available_props:
            raise ValueError("This descriptor is not available")

        if type(comp) == str:
            comp = Composition(comp)

        # Get data for given element/compound
        el_amt = comp.get_el_amt_dict()
        # sort symbols by electronegativity
        symbols = sorted(el_amt.keys(), key=lambda sym: get_el_sp(sym).Z)

        demldata = []

        if property_name == "formal_charge":
            fml_charge_dict = self.calc_formal_charge(comp)
            for el in symbols:
                try:
                    prop = float(fml_charge_dict[el])
                except:
                    prop = float("NaN")
                if combine_by_element:
                    demldata.append(prop)
                else:
                    for _ in range(int(el_amt[el])):
                        demldata.append(prop)
        elif property_name == "first_ioniz":  # First ionization energy
            for el in symbols:
                try:
                    first_ioniz = self.all_props["ionization_en"][el][0]
                except:
                    first_ioniz = float("NaN")
                if combine_by_element:
                    demldata.append(first_ioniz)
                else:
                    for _ in range(int(el_amt[el])):
                        demldata.append(first_ioniz)
        elif property_name == "total_ioniz":  # Cumulative ionization energy
            for el in symbols:
                try:
                    total_ioniz = sum(self.all_props["ionization_en"][el])
                except:
                    total_ioniz = float("NaN")
                if combine_by_element:
                    demldata.append(total_ioniz)
                else:
                    for _ in range(int(el_amt[el])):
                        demldata.append(total_ioniz)
        elif "valence" in property_name:
            for el in symbols:
                valence_dict = self.all_props["valence_e"][
                    self.all_props["col_num"][el]]
                if property_name[-1] in ["s", "p", "d"]:
                    if combine_by_element:
                        demldata.append(float(valence_dict[property_name[-1]]))
                    else:
                        for _ in range(int(el_amt[el])):
                            demldata.append(
                                float(valence_dict[property_name[-1]]))
                else:
                    n_valence = sum(valence_dict.values())
                    if combine_by_element:
                        demldata.append(float(n_valence))
                    else:
                        for _ in range(int(el_amt[el])):
                            demldata.append(float(n_valence))
        elif property_name in ["xtal_field_split", "magn_moment", "so_coupling",
                               "sat_magn"]:  # Charge dependent properties
            fml_charge_dict = self.calc_formal_charge(comp)
            for el in symbols:
                try:
                    charge = fml_charge_dict[el]
                    prop = float(self.all_props[property_name][el][charge])
                except:
                    prop = 0.0
                if combine_by_element:
                    demldata.append(prop)
                else:
                    for _ in range(int(el_amt[el])):
                        demldata.append(prop)
            return demldata
        else:
            for el in symbols:
                try:
                    prop = float(self.all_props[property_name][el])
                except:
                    prop = float("NaN")
                if combine_by_element:
                    demldata.append(prop)
                else:
                    for _ in range(int(el_amt[el])):
                        demldata.append(prop)

        return demldata


@singleton
class MagpieData(AbstractData):
    """
    Singleton class to get data from Magpie files
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

        self._parse()

    def _parse(self):
        """
        parse and store all elemental properties once and for all.
        """
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
                        str(Element.from_Z(atomic_no))] = prop_value

    def get_property(self, comp, property_name, combine_by_element=False):
        """
        Args:
            x: (comp) Composition object (or str representation)
            property_name (str):
            combine_by_element (bool):

        Returns:
            (list): list of property values for the composition
        """
        if property_name not in self.available_props:
            raise ValueError(
                "This descriptor is not available from the Magpie repository. "
                "Choose from {}".format(self.available_props))

        if type(comp) == str:
            comp = Composition(comp)

        # Get data for given element/compound
        el_amt = comp.get_el_amt_dict()
        # sort symbols by electronegativity
        symbols = sorted(el_amt.keys(), key=lambda sym: get_el_sp(sym).Z)

        if combine_by_element:
            return [self.all_elemental_props[property_name][el] for el in
                    symbols]
        else:
            return [self.all_elemental_props[property_name][el]
                    for el in symbols
                    for _ in range(int(el_amt[el]))]


class PymatgenData(AbstractData):
    def get_property(self, comp, property_name, combine_by_element=False):
        """
        Get descriptor data for elements in a compound from pymatgen.

        Args:
            comp (str/Composition): Either pymatgen Composition object or string formula,
                eg: "NaCl", "Na+1Cl-1", "Fe2+3O3-2" or "Fe2 +3 O3 -2"
                Notes:
                     - For 'ionic_radii' property, the Composition object must consist of oxidation
                        state decorated Specie objects not the plain Element objects.
                        eg.  fe2o3 = Composition({Specie("Fe", 3): 2, Specie("O", -2): 3})
                     - For string formula, the oxidation state sign(+ or -) must be specified explicitly.
                        eg.  "Fe2+3O3-2"

            property_name (str): pymatgen element attribute name, as defined in the Element class at
                http://pymatgen.org/_modules/pymatgen/core/periodic_table.html

            combine_by_element (bool):

        Returns:
            (list) of values containing descriptor floats for each atom in the compound (sorted by the
                atomic number of the constituent atoms)

        """
        eldata = []
        # what are these named tuples for? not used or returned! -KM
        eldata_tup_lst = []
        eldata_tup = namedtuple('eldata_tup',
                                'element propname propvalue propunit amt')

        oxidation_states = {}
        if isinstance(comp, Composition):
            # check whether the composition is composed of oxidation state decorated species
            if hasattr(comp.elements[0], "oxi_state"):
                oxidation_states = dict(
                    [(str(sp.element), sp.oxi_state) for sp in comp.elements])
            el_amt_dict = comp.get_el_amt_dict()
        # string
        else:
            comp, oxidation_states = self.get_composition_oxidation_state(comp)
            el_amt_dict = comp.get_el_amt_dict()

        symbols = sorted(el_amt_dict.keys(), key=lambda sym: get_el_sp(sym).Z)

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
                        property_value = element.ionic_radii[
                            oxidation_states[el_sym]]
                        property_units = Unit("ang")
                    else:
                        raise ValueError(
                            "oxidation state not given for {}; It does not yield a unique "
                            "number per Element".format(property_name))
                elif property_name == "block":
                    block_key = {"s": 1.0, "p": 2.0, "d": 3.0, "f": 3.0}
                    property_value = block_key[p]
                else:
                    property_value = float(p)

                # units are None for these pymatgen features
                # todo: there seem to be a lot more unitless features which are not listed here... -Alex D
                if property_name not in ['X', 'Z', 'group', 'row', 'number',
                                         'mendeleev_no',
                                         'ionic_radii', 'block']:
                    property_units = p.unit

            # Make a named tuple out of all the available information
            eldata_tup_lst.append(
                eldata_tup(element=el_sym, propname=property_name,
                           propvalue=property_value,
                           propunit=property_units, amt=el_amt_dict[el_sym]))

            if combine_by_element:
                if property_value is None:
                    eldata.append(float("NaN"))
                else:
                    eldata.append(property_value)
            else:
                # Add descriptor values, one for each atom in the compound
                if property_value is None:
                    eldata.append(float("NaN"))
                else:
                    for i in range(int(el_amt_dict[el_sym])):
                        eldata.append(property_value)

        return eldata

    @staticmethod
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
                oxidation_states.append(
                    "{}{}".format(sign[-1], digits[-1].strip()))
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
