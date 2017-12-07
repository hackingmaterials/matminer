from __future__ import division, unicode_literals, print_function

"""
Defines wrappers for data sources(magpi, pymatgen etc) for elemental properties.
"""

import os
import json
import six
import abc
import itertools
import numpy as np
from glob import glob
from collections import defaultdict, namedtuple

from monty.design_patterns import singleton

from pymatgen import Element, Composition, Unit
from pymatgen.core.periodic_table import _pt_data, get_el_sp

__author__ = 'Kiran Mathew, Jiming Chen, Logan Ward, Anubhav Jain'

module_dir = os.path.dirname(os.path.abspath(__file__))


class AbstractData((six.with_metaclass(abc.ABCMeta))):
    @abc.abstractmethod
    def get_property(self, x, property_name, *args, **kwargs):
        """
        Fetches data / information for an object x.

        Args:
            x: input data/object to retrieve the property of. Input type depends
                on type of abstract data, e.g., a Composition for MagpieData
            property_name (str): Name of property to retrieve
            *args: other arguments needed by the AbstractData object
            **kwargs: other keyword arguments needed by the AbstractData object

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

    def get_property(self, x, property_name="cohesive_energy"):
        """
        Args:
            x: (str) Element as str
            property_name (str): unused, always returns cohesive energy

        Returns:
            (float): cohesive energy of the element
        """
        return self.cohesive_energy_data[x]


@singleton
class DemlData(AbstractData):
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
            property_name (str): property to fetch, see self.available_props
                for list of possibilities
            combine_by_element (bool): If true, behavior will ignore
                stoichiometric ratios and will collapse all values for a single
                Element to one value (e.g., VO and VO2 will give the same data
                vector)

        Returns:
            (list): list of property values for the composition sorted by
                atomic number Z of each element
        """
        if property_name not in self.available_props:
            raise ValueError("This descriptor is not available")

        if type(comp) == str:
            comp = Composition(comp)

        # Get data for given element/compound
        el_amt = comp.get_el_amt_dict()
        # sort symbols by atomic number
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
                        str(Element.from_Z(atomic_no))] = prop_value

    def get_property(self, comp, property_name, combine_by_element=False):
        """
        Args:
            x: (comp) Composition object (or str representation)
            property_name (str): see self.available_props for a list of possibilities
            combine_by_element (bool): If true, behavior will ignore
                stoichiometric ratios and will collapse all values for a single
                Element to one value (e.g., VO and VO2 will give the same data
                vector)

        Returns:
            (list): list of property values for the composition sorted by
                atomic number Z of each element
        """
        if property_name not in self.available_props:
            raise ValueError(
                "This descriptor is not available from the Magpie repository. "
                "Choose from {}".format(self.available_props))

        if type(comp) == str:
            comp = Composition(comp)

        # Get data for given element/compound
        el_amt = comp.get_el_amt_dict()

        # sort symbols by Z
        symbols = sorted(el_amt.keys(), key=lambda sym: get_el_sp(sym).Z)

        if combine_by_element:
            return [self.all_elemental_props[property_name][el] for el in
                    symbols]
        else:
            return [self.all_elemental_props[property_name][el]
                    for el in symbols
                    for _ in range(int(el_amt[el]))]


class PymatgenData(AbstractData):
    """
    Class to get data from pymatgen. See also:
    S.P. Ong, W.D. Richards, A. Jain, G. Hautier, M. Kocher, S. Cholia, et al.,
    Python Materials Genomics (pymatgen): A robust, open-source python library
    for materials analysis, Comput. Mater. Sci. 68 (2013) 314-319.
    """

    def get_property(self, comp, property_name):
        """
        Get descriptor data for elements in a compound from pymatgen.

        Args:
            comp (str/Composition): pymatgen Composition object
            property_name (str): pymatgen element attribute name, as defined in
                the Element class at http://pymatgen.org/_modules/pymatgen/core/periodic_table.html.
                For 'ionic_radii' property, the Composition object must consist
                of oxidation state decorated Specie objects (not plain Element objects).

        Returns:
            (list) of values containing descriptor floats for each atom in the
                compound (sorted by the atomic number Z of the constituent atoms)

        """
        eldata = []
        # what are these named tuples for? not used or returned! -KM
        eldata_tup_lst = []
        eldata_tup = namedtuple('eldata_tup',
                                'element propname propvalue propunit amt')

        el_amt_dict = comp.get_el_amt_dict()

        # check whether the composition is composed of oxidation state
        # decorated species
        oxidation_states = {}
        has_oxi_state = all([hasattr(el, "oxi_state") for el in comp])

        if has_oxi_state:
            oxidation_states = dict(
                [(str(sp.element), sp.oxi_state) for sp in comp.elements])

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
                if property_name == 'ionic_radii':
                    if oxidation_states:
                        property_value = element.ionic_radii[
                            oxidation_states[el_sym]]
                        property_units = Unit("ang")
                    else:
                        raise ValueError(
                            "ionic_radii specified but oxidation state not given "
                            "for {}".format(property_name))

                elif property_name == "block":
                    block_key = {"s": 1.0, "p": 2.0, "d": 3.0, "f": 3.0}
                    property_value = block_key[p]

                else:
                    property_value = float(p)

                property_units = getattr(p, "unit", None)

            # Make a named tuple out of all the available information
            eldata_tup_lst.append(
                eldata_tup(element=el_sym, propname=property_name,
                           propvalue=property_value,
                           propunit=property_units, amt=el_amt_dict[el_sym]))

            # Add descriptor values, one for each atom in the compound
            if property_value is None:
                eldata.append(float("NaN"))
            else:
                for i in range(int(el_amt_dict[el_sym])):
                    eldata.append(property_value)

        return eldata
