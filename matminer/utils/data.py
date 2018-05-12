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
import numpy as np
import pandas as pd
from glob import glob

from pymatgen import Element
from pymatgen.core.periodic_table import _pt_data

__author__ = 'Kiran Mathew, Jiming Chen, Logan Ward, Anubhav Jain'

module_dir = os.path.dirname(os.path.abspath(__file__))


class AbstractData(six.with_metaclass(abc.ABCMeta)):
    """Abstract class for retrieving elemental properties

    All classes must implement the `get_elemental_property` operation. These operations
    should return scalar values (ideally floats) and `nan` if a property does not exist"""

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
        """Retrieve the possible oxidation states of an element

        Args:
            elem - (Element), Target element
        Returns:
            [int] - oxidation states
        """
        pass


class OxidationStateDependentData(AbstractData):
    """Abstract class that also includes oxidation-state-dependent properties"""

    @abc.abstractmethod
    def get_charge_dependent_property(self, element, charge, property_name):
        """Retrieve a oxidation-state dependent elemental property

        Args:
            element - (Element), Target element
            charge - (int), Oxidation state
            property_name - (string), name of property
        Return:
            (float) - Value of property
        """
        pass

    def get_charge_dependent_property_from_specie(self, specie, property_name):
        """Retrieve a oxidation-state dependent elemental property

        Args:
            specie - (Specie), Specie of interest
            property_name - (string), name of property
        Return:
            (float) - Value of property
        """

        return self.get_charge_dependent_property(specie.element, specie.oxi_state, property_name)


class CohesiveEnergyData(AbstractData):
    """Get the cohesive energy of an element.

    Data is extracted from KnowledgeDoor Cohesive Energy Handbook online
    (http://www.knowledgedoor.com/2/elements_handbook/cohesive_energy.html),
    which in turn got the data from Introduction to Solid State Physics,
    8th Edition, by Charles Kittel (ISBN 978-0-471-41526-8), 2005.
    """

    def __init__(self):
        # Load elemental cohesive energy data from json file
        with open(os.path.join(module_dir, 'data_files',
                               'cohesive_energies.json'), 'r') as f:
            self.cohesive_energy_data = json.load(f)

    def get_elemental_property(self, elem, property_name='cohesive energy'):
        """
        Args:
            elem: (Element) Element of interest
            property_name (str): unused, always returns cohesive energy

        Returns:
            (float): cohesive energy of the element
        """
        return self.cohesive_energy_data[elem]


class DemlData(OxidationStateDependentData, OxidationStatesMixin):
    """
    Class to get data from Deml data file. See also: A.M. Deml,
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

        # Compute the FERE correction energy
        fere_corr = {}
        for k, v in self.all_props["GGAU_Etot"].items():
            fere_corr[k] = self.all_props["mus_fere"][k] - v
        self.all_props["FERE correction"] = fere_corr

        # List out the available charge-dependent properties
        self.charge_dependent_properties = ["xtal_field_split", "magn_moment", "so_coupling", "sat_magn"]

    def get_elemental_property(self, elem, property_name):
        if "valence" in property_name:
            valence_dict = self.all_props["valence_e"][
                self.all_props["col_num"][elem.symbol]]
            if property_name[-1] in ["s", "p", "d"]:
                # Return one of the shells
                return valence_dict[property_name[-1]]
            else:
                return sum(valence_dict.values())
        elif property_name == "first_ioniz":
            return self.all_props["ionization_en"][elem.symbol][0]
        else:
            return self.all_props[property_name].get(elem.symbol, float("NaN"))

    def get_oxidation_states(self, elem):
        return self.all_props["charge_states"][elem.symbol]

    def get_charge_dependent_property(self, element, charge, property_name):
        if property_name == "total_ioniz":
            if charge < 0:
                raise ValueError("total ionization energy only defined for charge > 0")
            return sum(self.all_props["ionization_en"][element.symbol][:charge])
        else:
            return self.all_props[property_name].get(element.symbol, {}).get(charge, np.nan)


class MagpieData(AbstractData, OxidationStatesMixin):
    """
    Class to get data from Magpie files. See also:
    L. Ward, A. Agrawal, A. Choudhary, C. Wolverton, A general-purpose machine
    learning framework for predicting properties of inorganic materials,
    Npj Comput. Mater. 2 (2016) 16028.
    """

    def __init__(self):
        self.all_elemental_props = dict()
        available_props = []
        self.data_dir = os.path.join(module_dir, "data_files",
                                     'magpie_elementdata')

        # Make a list of available properties
        for datafile in glob(os.path.join(self.data_dir, "*.table")):
            available_props.append(
                os.path.basename(datafile).replace('.table', ''))

        # parse and store elemental properties
        for descriptor_name in available_props:
            with open(os.path.join(self.data_dir,
                                   '{}.table'.format(descriptor_name)),
                      'r') as f:
                self.all_elemental_props[descriptor_name] = dict()
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


class PymatgenData(OxidationStateDependentData, OxidationStatesMixin):
    """
    Class to get data from pymatgen. See also:
    S.P. Ong, W.D. Richards, A. Jain, G. Hautier, M. Kocher, S. Cholia, et al.,
    Python Materials Genomics (pymatgen): A robust, open-source python library
    for materials analysis, Comput. Mater. Sci. 68 (2013) 314-319.
    """

    def __init__(self, use_common_oxi_states=True):
        self.use_common_oxi_states = use_common_oxi_states

    def get_elemental_property(self, elem, property_name):
        if property_name == "block":
            block_key = {"s": 1.0, "p": 2.0, "d": 3.0, "f": 3.0}
            return block_key[getattr(elem, property_name)]
        else:
            value = getattr(elem, property_name)
            return np.nan if value is None else value

    def get_oxidation_states(self, elem):
        """Get the oxidation states of an element

        Args:
            elem - (Element) target element
            common - (boolean), whether to return only the common oxidation states,
                or all known oxidation states
        Returns:
            [int] list of oxidation states
            """
        return elem.common_oxidation_states if self.use_common_oxi_states \
            else elem.oxidation_states

    def get_charge_dependent_property(self, element, charge, property_name):
        return getattr(element, property_name)[charge]


class MixingEnthalpy:
    """
    Values of :math:`\Delta H^{max}_{AB}` for different pairs of elements.

    Based on the Miedema model. Tabulated by:
        A. Takeuchi, A. Inoue, Classification of Bulk Metallic Glasses by Atomic
        Size Difference, Heat of Mixing and Period of Constituent Elements and
        Its Application to Characterization of the Main Alloying Element.
        Mater. Trans. 46, 2817–2829 (2005).
    """

    def __init__(self):
        mixing_dataset = pd.read_csv(os.path.join(module_dir, 'data_files',
                                                  'MiedemaLiquidDeltaHf.tsv'),
                                     delim_whitespace=True)
        self.mixing_data = {}
        for a, b, dHf in mixing_dataset.itertuples(index=False):
            key = tuple(sorted((a, b)))
            self.mixing_data[key] = dHf

    def get_mixing_enthalpy(self, elemA, elemB):
        """
        Get the mixing enthalpy between different elements

        Args:
            elemA (Element): An element
            elemB (Element): Second element
        Returns:
            (float) mixing enthalpy, nan if pair is not in a table
        """

        key = tuple(sorted((elemA.symbol, elemB.symbol)))
        return self.mixing_data.get(key, np.nan)
