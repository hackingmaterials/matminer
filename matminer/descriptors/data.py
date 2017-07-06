from __future__ import division, unicode_literals, print_function

import os
import json
from glob import glob
from collections import defaultdict

from monty.design_patterns import singleton

from pymatgen import Element, Composition
from pymatgen.core.periodic_table import _pt_data, get_el_sp

__author__ = 'Jimin Chen, Logan Ward, Kiran Mathew'

module_dir = os.path.dirname(os.path.abspath(__file__))

# Load elemental cohesive energy data from json file
with open(os.path.join(module_dir, 'data_files','cohesive_energies.json'), 'r') as f:
    cohesive_energy_data = json.load(f)


@singleton
class MagpieData:
    """
    Singleton class to get data from Magpie files
    """

    def __init__(self):
        self.all_elemental_props = defaultdict(dict)
        self.available_props = []
        self.data_dir = os.path.join(module_dir, "data_files", 'magpie_elementdata')

        # Make a list of available properties
        for datafile in glob(os.path.join(self.data_dir, "*.table")):
            self.available_props.append(os.path.basename(datafile).replace('.table', ''))

        self._parse()

    def _parse(self):
        """
        parse and store all elemental properties once and for all.
        """
        for descriptor_name in self.available_props:
            with open(os.path.join(self.data_dir, '{}.table'.format(descriptor_name)), 'r') as f:
                lines = f.readlines()
                for atomic_no in range(1, len(_pt_data)+1):  # max Z=103
                    try:
                        if descriptor_name in ["OxidationStates"]:
                            prop_value = [float(i) for i in lines[atomic_no - 1].split()]
                        else:
                            prop_value = float(lines[atomic_no - 1])
                    except ValueError:
                        prop_value = float("NaN")
                    self.all_elemental_props[descriptor_name][str(Element.from_Z(atomic_no))] = prop_value

    def get_data(self, comp, descriptor_name):
        """
        Gets magpie data for a composition object.

        Args:
            comp (Composition/str): Pymatgen composition object or str.
            descriptor_name (str): Name of descriptor

        Returns:
            magpiedata (list): list of values for each atom in comp_obj
        """
        comp = Composition(comp)
        if descriptor_name not in self.available_props:
            raise ValueError("This descriptor is not available from the Magpie repository. "
                             "Choose from {}".format(self.available_props))

        # Get data for given element/compound
        el_amt = comp.get_el_amt_dict()
        # sort symbols by electronegativity
        symbols = sorted(el_amt.keys(), key=lambda sym: get_el_sp(sym).X)

        return [self.all_elemental_props[descriptor_name][el]
                for el in symbols
                for _ in range(int(el_amt[el]))]


magpie_data = MagpieData()