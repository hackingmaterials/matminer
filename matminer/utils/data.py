"""
Utility classes for retrieving elemental properties. Provides
a uniform interface to several different elemental property resources
including ``pymatgen`` and ``Magpie``.
"""

import abc
import json
import os
import tarfile
import warnings
from copy import deepcopy
from glob import glob

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from pymatgen.core.periodic_table import Element
from ruamel import yaml
from scipy import stats
from scipy.interpolate import interp1d

from matminer.utils.utils import get_elem_in_data, get_pseudo_inverse
from matminer.utils.warnings import IMPUTE_NAN_WARNING

__author__ = "Kiran Mathew, Jiming Chen, Logan Ward, Anubhav Jain, Alex Dunn"

module_dir = os.path.dirname(os.path.abspath(__file__))


class AbstractData(metaclass=abc.ABCMeta):
    """Abstract class for retrieving elemental properties

    All classes must implement the `get_elemental_property` operation. These operations
    should return scalar values (ideally floats) and `nan` if a property does not exist"""

    @abc.abstractmethod
    def get_elemental_property(self, elem, property_name):
        """Get a certain elemental property for a certain element.

        Args:
            elem - (Element) element to be assessed
            property_name - (str) property to be retrieved
        Returns:
            float, property of that element
        """

    def get_elemental_properties(self, elems, property_name):
        """Get elemental properties for a list of elements

        Args:
            elems - ([Element]) list of elements
            property_name - (str) property to be retrieved
        Returns:
            [float], properties of elements
        """
        return [self.get_elemental_property(e, property_name) for e in elems]


class OxidationStatesMixin(metaclass=abc.ABCMeta):
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

    Args:
        impute_nan (bool): if True, the features for the elements
            that are missing from the data_source or are NaNs are replaced by the
            average of each features over the available elements.
    """

    def __init__(self, impute_nan=False):
        # Load elemental cohesive energy data from json file
        with open(os.path.join(module_dir, "data_files", "cohesive_energies.json")) as f:
            self.cohesive_energy_data = json.load(f)

        self.impute_nan = impute_nan
        if self.impute_nan:
            elem_list = list(self.cohesive_energy_data)
            missing_elements = [e.symbol for e in Element if e.symbol not in elem_list]
            avg_cohesive_energy = np.nanmean(list(self.cohesive_energy_data.values()))

            for e in Element:
                if e.symbol in missing_elements or np.isnan(self.cohesive_energy_data[e.symbol]):
                    self.cohesive_energy_data[e.symbol] = avg_cohesive_energy

        else:
            warnings.warn(f"{self.__class__.__name__}(impute_nan=False):\n" + IMPUTE_NAN_WARNING)

    def get_elemental_property(self, elem, property_name="cohesive energy"):
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

    The meanings of each feature in the data can be found in
    ./data_files/deml_elementdata.py

    Args:
        impute_nan (bool): if True, the features for the elements
            that are missing from the data_source or are NaNs are replaced by the
            average of each features over the available elements.
    """

    def __init__(self, impute_nan=False):
        from matminer.utils.data_files.deml_elementdata import properties

        self.all_props = deepcopy(properties)
        self.available_props = list(self.all_props.keys()) + [
            "formal_charge",
            "valence_s",
            "valence_p",
            "valence_d",
            "first_ioniz",
            "total_ioniz",
        ]

        # List out the available charge-dependent properties
        self.charge_dependent_properties = [
            "xtal_field_split",
            "magn_moment",
            "so_coupling",
            "sat_magn",
        ]

        self.impute_nan = impute_nan
        if self.impute_nan:
            avg_ioniz = np.nanmean([self.all_props["ionization_en"].get(e.symbol, [float("NaN")])[0] for e in Element])

            for e in Element:
                if e.symbol not in self.all_props["atom_num"]:
                    self.all_props["atom_num"][e.symbol] = e.Z
                if e.symbol not in self.all_props["atom_mass"]:
                    self.all_props["atom_mass"][e.symbol] = e.atomic_mass
                if e.symbol not in self.all_props["row_num"]:
                    self.all_props["row_num"][e.symbol] = e.row
                if e.symbol not in self.all_props["ionization_en"] or np.isnan(
                    self.all_props["ionization_en"][e.symbol][0]
                ):
                    self.all_props["ionization_en"][e.symbol] = [avg_ioniz]
                for key in [
                    "col_num",
                    "atom_radius",
                    "molar_vol",
                    "heat_fusion",
                    "melting_point",
                    "boiling_point",
                    "heat_cap",
                    "electron_affin",
                    "electronegativity",
                    "electric_pol",
                    "GGAU_Etot",
                    "mus_fere",
                    "electron_affin",
                ]:
                    if e.symbol not in self.all_props[key] or np.isnan(self.all_props[key][e.symbol]):
                        self.all_props[key][e.symbol] = np.nanmean(list(self.all_props[key].values()))
        else:
            warnings.warn(f"{self.__class__.__name__}(impute_nan=False):\n" + IMPUTE_NAN_WARNING)

        # Compute the FERE correction energy
        fere_corr = {}
        for k, v in self.all_props["GGAU_Etot"].items():
            fere_corr[k] = self.all_props["mus_fere"][k] - v
        self.all_props["FERE correction"] = fere_corr

    def get_elemental_property(self, elem, property_name):
        if "valence" in property_name:
            valence_dict = self.all_props["valence_e"][self.all_props["col_num"][elem.symbol]]
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
        if not self.impute_nan:
            return self.all_props["charge_states"][elem.symbol]
        else:
            return self.all_props["charge_states"].get(elem.symbol, [0])

    def get_charge_dependent_property(self, element, charge, property_name):
        if property_name == "total_ioniz":
            if charge < 0:
                raise ValueError("total ionization energy only defined for charge > 0")
            return sum(self.all_props["ionization_en"][element.symbol][:charge])
        elif self.impute_nan:
            return self.all_props[property_name].get(element.symbol, {}).get(charge, 0)
        else:
            return self.all_props[property_name].get(element.symbol, {}).get(charge, np.nan)


class MagpieData(AbstractData, OxidationStatesMixin):
    """
    Class to get data from Magpie files. See also:
    L. Ward, A. Agrawal, A. Choudhary, C. Wolverton, A general-purpose machine
    learning framework for predicting properties of inorganic materials,
    Npj Comput. Mater. 2 (2016) 16028.


    Finding the exact meaning of each of these features can be quite difficult.
    Reproduced in ./data_files/magpie_elementdata_feature_descriptions.txt.

    Args:
        impute_nan (bool): if True, the features for the elements
            that are missing from the data_source or are NaNs are replaced by the
            average of each features over the available elements.
    """

    def __init__(self, impute_nan=False):
        self.all_elemental_props = dict()
        available_props = []
        self.data_dir = os.path.join(module_dir, "data_files", "magpie_elementdata")

        # Make a list of available properties
        for datafile in glob(os.path.join(self.data_dir, "*.table")):
            available_props.append(os.path.basename(datafile).replace(".table", ""))

        # parse and store elemental properties
        for descriptor_name in available_props:
            with open(os.path.join(self.data_dir, f"{descriptor_name}.table")) as f:
                self.all_elemental_props[descriptor_name] = dict()
                lines = f.readlines()
                for atomic_no in range(1, 118 + 1):  # (max Z=118)
                    try:
                        if descriptor_name in ["OxidationStates"]:
                            prop_value = [float(i) for i in lines[atomic_no - 1].split()]
                        else:
                            prop_value = float(lines[atomic_no - 1])
                    except (ValueError, IndexError):
                        prop_value = float("NaN")
                    self.all_elemental_props[descriptor_name][Element.from_Z(atomic_no).symbol] = prop_value

        self.impute_nan = impute_nan
        if self.impute_nan:
            for prop in self.all_elemental_props:
                if prop == "OxidationStates":
                    nested_props = list(self.all_elemental_props["OxidationStates"].values())
                    flatten_props = []
                    for l in nested_props:
                        if isinstance(l, list):
                            for e in l:
                                flatten_props.append(e)
                        else:
                            flatten_props.append(l)

                    avg_prop = np.round(np.nanmean(flatten_props))
                    for e in Element:
                        if (
                            e.symbol not in self.all_elemental_props[prop]
                            or self.all_elemental_props[prop][e.symbol] == []
                            or np.any(np.isnan(self.all_elemental_props[prop][e.symbol]))
                        ):
                            self.all_elemental_props[prop][e.symbol] = [avg_prop]
                else:
                    avg_prop = np.nanmean(list(self.all_elemental_props[prop].values()))
                    for e in Element:
                        if e.symbol not in self.all_elemental_props[prop] or np.isnan(
                            self.all_elemental_props[prop][e.symbol]
                        ):
                            self.all_elemental_props[prop][e.symbol] = avg_prop
        else:
            warnings.warn(f"{self.__class__.__name__}(impute_nan=False):\n" + IMPUTE_NAN_WARNING)

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

    Meanings of each feature can be obtained from the pymatgen.Composition
    documentation (attributes).

    Args:
        impute_nan (bool): if True, the features for the elements
            that are missing from the data_source or are NaNs are replaced by the
            average of each features over the available elements.
    """

    def __init__(self, use_common_oxi_states=True, impute_nan=False):
        self.use_common_oxi_states = use_common_oxi_states
        self.impute_nan = impute_nan

    def get_elemental_property(self, elem, property_name):
        if property_name == "block":
            block_key = {"s": 1.0, "p": 2.0, "d": 3.0, "f": 3.0}
            return block_key[getattr(elem, property_name)]
        else:
            # Suppress pymatgen warnings for missing properties if we are going to impute anyway
            with warnings.catch_warnings():
                if self.impute_nan:
                    warnings.simplefilter("ignore", category=UserWarning)
                value = getattr(elem, property_name)

            if self.impute_nan:
                if value and not pd.isnull(value):
                    return value
                else:
                    if property_name == "ionic_radii":
                        return {0: self.get_elemental_property(elem, "atomic_radius")}
                    elif property_name in ["common_oxidation_states", "icsd_oxidation_states"]:
                        return (0,)
                    else:
                        all_values = [getattr(e, property_name) for e in Element]
                        all_values = [val if val else float("NaN") for val in all_values]
                        value_avg = np.nanmean(all_values)
                    return value_avg
            else:
                warnings.warn(f"{self.__class__.__name__}(impute_nan=False):\n" + IMPUTE_NAN_WARNING)
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
        return elem.common_oxidation_states if self.use_common_oxi_states else elem.oxidation_states

    def get_charge_dependent_property(self, element, charge, property_name):
        if self.impute_nan:
            if property_name == "ionic_radii":
                return self.get_elemental_property(element, property_name)[charge]
            elif property_name in ["common_oxidation_states", "icsd_oxidation_states"]:
                return self.get_elemental_property(element, property_name)[charge]
        else:
            return getattr(element, property_name)[charge]


class MixingEnthalpy:
    r"""
    Values of :math:`\Delta H^{max}_{AB}` for different pairs of elements.

    Based on the Miedema model. Tabulated by:
        A. Takeuchi, A. Inoue, Classification of Bulk Metallic Glasses by Atomic
        Size Difference, Heat of Mixing and Period of Constituent Elements and
        Its Application to Characterization of the Main Alloying Element.
        Mater. Trans. 46, 2817–2829 (2005).

    Attributes:
        valid_element_list ([Element]): A list of elements for which the
            mixing enthalpy parameters are defined (although no guarantees
            are provided that all combinations of this list will be available).

    Args:
        impute_nan (bool): if True, the features for the elements
            that are missing from the data_source or are NaNs are replaced by the
            average of each features over the available elements.
    """

    def __init__(self, impute_nan=False):
        mixing_dataset = pd.read_csv(
            os.path.join(module_dir, "data_files", "MiedemaLiquidDeltaHf.tsv"),
            sep=r"\s+",
        )
        self.mixing_data = {}
        for a, b, dHf in mixing_dataset.itertuples(index=False):
            key = tuple(sorted((a, b)))
            self.mixing_data[key] = dHf
        valid_elements = [
            "Dy",
            "Mn",
            "Y",
            "Nd",
            "Ag",
            "Cs",
            "Tm",
            "Pd",
            "Sn",
            "Rh",
            "Pr",
            "Er",
            "K",
            "In",
            "Tb",
            "Rb",
            "H",
            "N",
            "Ni",
            "Hg",
            "Ca",
            "Mo",
            "Li",
            "Th",
            "U",
            "At",
            "Ga",
            "La",
            "Ru",
            "Lu",
            "Eu",
            "Si",
            "B",
            "Zr",
            "Ce",
            "Pm",
            "Ge",
            "Sm",
            "Ta",
            "Ti",
            "Po",
            "Sc",
            "Mg",
            "Sr",
            "P",
            "C",
            "Ir",
            "Pa",
            "V",
            "Zn",
            "Sb",
            "Na",
            "W",
            "Re",
            "Tl",
            "Pt",
            "Gd",
            "Cr",
            "Co",
            "Ba",
            "Os",
            "Hf",
            "Pb",
            "Cu",
            "Tc",
            "Al",
            "As",
            "Ho",
            "Yb",
            "Au",
            "Be",
            "Nb",
            "Cd",
            "Fe",
            "Bi",
        ]
        self.valid_element_list = [Element(e) for e in valid_elements]

        self.impute_nan = impute_nan
        if self.impute_nan:
            avg_value = np.nanmean(list(self.mixing_data.values()))
            for e1 in Element:
                for e2 in Element:
                    key = tuple(sorted((e1.symbol, e2.symbol)))
                    if key not in self.mixing_data or np.isnan(self.mixing_data[key]):
                        self.mixing_data[key] = avg_value
            self.valid_element_list = list(Element)
        else:
            warnings.warn(f"{self.__class__.__name__}(impute_nan=False):\n" + IMPUTE_NAN_WARNING)

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


class MatscholarElementData(AbstractData):
    """
    Class to get word embedding vectors of elements. These word embeddings were
    generated using NLP + Neural Network techniques on more than 3 million
    scientific abstracts.

    The data returned by this class are simply learned representations of the
    elements, taken from:

    Tshitoyan, V., Dagdelen, J., Weston, L. et al. Unsupervised word embeddings
    capture latent knowledge from materials science literature. Nature 571,
    95–98 (2019). https://doi.org/10.1038/s41586-019-1335-8

    Args:
        impute_nan (bool): if True, the features for the elements
            that are missing from the data_source or are NaNs are replaced by the
            average of each features over the available elements.
    """

    def __init__(self, impute_nan=False):
        dfile = os.path.join(module_dir, "data_files/matscholar_els.json")
        with open(dfile) as fp:
            embeddings = json.load(fp)
        self.prop_names = [f"embedding {i}" for i in range(1, 201)]
        all_element_data = {}
        for el, embedding in embeddings.items():
            all_element_data[el] = dict(zip(self.prop_names, embedding))
        self.all_element_data = all_element_data

        self.impute_nan = impute_nan
        if self.impute_nan:
            for embedding in self.prop_names:
                avg_value = np.nanmean(
                    [self.all_element_data[e].get(embedding, float("NaN")) for e in self.all_element_data]
                )
                for e in Element:
                    if e.symbol not in self.all_element_data:
                        self.all_element_data[e.symbol] = {embedding: avg_value}
                    elif embedding not in self.all_element_data[e.symbol] or np.isnan(
                        self.all_element_data[e.symbol][embedding]
                    ):
                        self.all_element_data[e.symbol][embedding] = avg_value
        else:
            warnings.warn(f"{self.__class__.__name__}(impute_nan=False):\n" + IMPUTE_NAN_WARNING)

    def get_elemental_property(self, elem, property_name):
        return self.all_element_data[str(elem)][property_name]


class MEGNetElementData(AbstractData):
    """
    Class to get neural network embeddings of elements. These embeddings were
    generated using the Materials Graph Network (MEGNet) developed by the
    MaterialsVirtualLab at U.C. San Diego and described in the publication:

    Graph Networks as a Universal Machine Learning Framework for Molecules and
    Crystals. Chi Chen, Weike Ye, Yunxing Zuo, Chen Zheng, and Shyue Ping Ong
    Chemistry of Materials 2019 31 (9), 3564-3572,
    https://doi.org/10.1021/acs.chemmater.9b01294

    The code for MEGNet can be found at:
    https://github.com/materialsvirtuallab/megnet

    The embeddings were generated by training the MEGNet Graph Network on
    60,000 structures from the Materials Project for predicting formation
    energy, and may be an effective way of applying transfer learning to
    smaller datasets using crystal-graph-based networks.

    The representations are learned during training to predict a specific
    property, though they may be useful for a range of properties.

    Args:
        impute_nan (bool): if True, the features for the elements
            that are missing from the data_source or are NaNs are replaced by the
            average of each features over the available elements.
    """

    def __init__(self, impute_nan=False):
        dfile = os.path.join(module_dir, "data_files/megnet_elemental_embedding.json")
        self._dummy = "Dummy"
        with open(dfile) as fp:
            embeddings = json.load(fp)
        self.prop_names = [f"embedding {i}" for i in range(1, 17)]
        self.all_element_data = {}
        for i in range(95):
            embedding_dict = dict(zip(self.prop_names, embeddings[i]))
            if i == 0:
                self.all_element_data[self._dummy] = embedding_dict
            else:
                self.all_element_data[str(Element.from_Z(i))] = embedding_dict

        self.impute_nan = impute_nan
        if self.impute_nan:
            for embedding in self.prop_names:
                avg_value = np.nanmean(
                    [self.all_element_data[e].get(embedding, float("NaN")) for e in self.all_element_data]
                )
                for e in Element:
                    if e.symbol not in self.all_element_data:
                        self.all_element_data[e.symbol] = {embedding: avg_value}
                    elif embedding not in self.all_element_data[e.symbol] or np.isnan(
                        self.all_element_data[e.symbol][embedding]
                    ):
                        self.all_element_data[e.symbol][embedding] = avg_value
        else:
            warnings.warn(f"{self.__class__.__name__}(impute_nan=False):\n" + IMPUTE_NAN_WARNING)

    def get_elemental_property(self, elem, property_name):
        estr = str(elem)
        if estr not in self.all_element_data.keys():
            estr = self._dummy
        return self.all_element_data[estr][property_name]


class IUCrBondValenceData:
    """Get empirical bond valence parameters.

    Data come from International Union of Crystallography 2016 tables.
    (https://www.iucr.org/resources/data/datasets/bond-valence-parameters)
    Both the raw source CIF and cleaned csv file are made accessible here.
    Within the source CIF, there are citations for every set of parameters.

    The copyright notice and disclaimer are reproduced below
    #***************************************************************
    # COPYRIGHT NOTICE
    # This table may be used and distributed without fee for
    # non-profit purposes providing
    # 1) that this copyright notice is included and
    # 2) no fee is charged for the table and
    # 3) details of any changes made in this list by anyone other than
    # the copyright owner are suitably noted in the _audit_update record
    # Please consult the copyright owner regarding any other uses.
    #
    # The copyright is owned by I. David Brown, Brockhouse Institute for
    # Materials Research, McMaster University, Hamilton, Ontario Canada.
    # idbrown@mcmaster.ca
    #
    #*****************************DISCLAIMER************************
    #
    # The values reported here are taken from the literature and
    # other sources and the author does not warrant their correctness
    # nor accept any responsibility for errors.  Users are advised to
    # consult the primary sources.
    #
    #***************************************************************
    """

    def __init__(self, interpolate_soft=True):
        """
        Load bond valence parameters as pandas dataframe.

        If interpolate_soft is True, fill in some missing values
        for anions such as I, Br, N, S, Se, etc. with the assumption
        that bond valence parameters of such anions don't depend on
        cation oxidation state. This assumption comes from Brese and O'Keeffe,
        (1991), Acta Cryst. B47, 194, which states "with less electronegative
        anions, ... R is not very different for different oxidation states in
        general." In the original data source file, only one set of parameters
        is usually provided for those less electronegative anions in a 9+
        oxidation state, indicating they can be used with all oxidation states.
        """
        filepath = os.path.join(module_dir, "data_files", "bvparm2020.cif")
        self.params: pd.DataFrame = pd.read_csv(
            filepath,
            sep=r"\s+",
            header=None,
            names=[
                "Atom1",
                "Atom1_valence",
                "Atom2",
                "Atom2_valence",
                "Ro",
                "B",
                "ref_id",
                "details",
            ],
            skiprows=172,
            skipfooter=1,
            index_col=False,
            engine="python",
        )
        if interpolate_soft:
            self.params = self.interpolate_soft_anions()

    def interpolate_soft_anions(self):
        """Fill in missing parameters for oxidation states of soft anions."""
        high_electroneg = "|".join(["O", "Cl", "F"])
        has_high = self.params["Atom2"].str.contains(high_electroneg)
        has_high[pd.isnull(has_high)] = False
        subset = self.params.loc[(self.params["Atom1_valence"] == 9) & (~has_high)]
        cation_subset = subset["Atom1"].unique()
        data = []
        for cation in cation_subset:
            anions = subset.loc[subset["Atom1"] == cation]["Atom2"].unique()
            for anion in anions:
                an_val, Ro, b, ref_id = subset.loc[(subset["Atom1"] == cation) & (subset["Atom2"] == anion)][
                    ["Atom2_valence", "Ro", "B", "ref_id"]
                ].values[0]
                for n in range(1, 7):
                    entry = {
                        "Atom1": cation,
                        "Atom1_valence": n,
                        "Atom2": anion,
                        "Atom2_valence": an_val,
                        "Ro": Ro,
                        "B": b,
                        "ref_id": ref_id,
                        "details": "Interpolated",
                    }
                    data.append(entry)
        new_data = pd.DataFrame(data)
        new_params = pd.concat((self.params, new_data), sort=True, ignore_index=True)
        return new_params

    def get_bv_params(self, cation, anion, cat_val, an_val):
        """Lookup bond valence parameters from IUPAC table.
        Args:
            cation (Element): cation element
            anion (Element): anion element
            cat_val (Integer): cation formal oxidation state
            an_val (Integer): anion formal oxidation state
        Returns:
            bond_val_list: dataframe of bond valence parameters
        """

        bv_data = self.params
        bond_val_list = self.params.loc[
            (bv_data["Atom1"] == str(cation))
            & (bv_data["Atom1_valence"] == cat_val)
            & (bv_data["Atom2"] == str(anion))
            & (bv_data["Atom2_valence"] == an_val)
        ]
        return bond_val_list.iloc[0]  # If multiple values exist, take first one
        # as recommended for reliability.


class OpticalData(AbstractData):
    """
    Class to use optical data from https://www.refractiveindex.info
    The properties are the refractive index n, the extinction coefficient ĸ
    (measured or computed with DFT), and the reflectivity R as obtained from
    Fresnel's equation.
    Data is by default considered if available from 380 to 780 nm,
    but other ranges can be chosen as well.

    In case new data becomes available and needs to be added to the database,
    it should be added in matminer/utils/data_files/optical_polyanskiy/database,
    which should then be compressed in the tar.xz format.
    To add a file for a compound, follow any of the formats of refractiveindex.info.

    The database is used to extract:
    1) the properties of single elements when available.
    2) the pseudo-inverse of the properties of single elements,
       based on the data for ~200 compounds. These pseudo-inverse
       contributions correspond to the coefficients of a least-square fit
       from the compositions to the properties. This can allow to better take into account
       data from different compounds for a given element.

    Using the pseudo-inverses (method="pseudo_inverse") instead of
    the elemental properties (method="exact") leads to better results as far as we have checked.
    Another possibility is to use method="combined", where the exact values are taken
    for compounds present as pure compounds in the database, and the pseudo-inverse
    is taken if the element is not present purely in the database.

    n, ĸ, and R are spectra. These are composed of n_wl wavelengths, from min_wl to max_wl.
    We split these spectra into bins (initially 10) where their average values are taken.
    These averaged values are the final features. The wavelength corresponding to a given bin
    is its midpoint.

    Args:
        props: optical properties to include. Should be a list with
               "refractive" and/or "extinction" and/or "reflectivity".
        method: type of values, either "exact", "pseudo_inverse", or "combined".
        min_wl: minimum wavelength to include in the spectra (µm).
        max_wl : maximum wavelength to include in the spectra (µm).
        n_wl: number of wavelengths to include in the spectra.
        bins: number of bins to split the spectra.
        saving_dir: folder to save the data and csv file used for the featurization. Saving them helps fasten the
                    featurization.
        impute_nan (bool): if True, the features for the elements
            that are missing from the data_source or are NaNs are replaced by the
            average of each features over the available elements.
    """

    def __init__(
        self,
        props=None,
        method="pseudo_inverse",
        min_wl=0.38,
        max_wl=0.78,
        n_wl=401,
        bins=10,
        saving_dir="~/.matminer/optical_props/",
        impute_nan=False,
    ):
        # Handles the saving folder
        saving_dir = os.path.expanduser(saving_dir)
        os.makedirs(os.path.join(saving_dir, "database"), exist_ok=True)
        self.saving_dir = saving_dir

        # Handle the selection of properties
        if props is None:
            props = ["refractive", "extinction", "reflectivity"]
        elif not all([prop in ["refractive", "extinction", "reflectivity"] for prop in props]):
            raise ValueError("This property is not available: choose from refractive, extinction, or reflectivity")

        self.props = props
        self.method = method
        self.n_wl = n_wl
        self.min_wl = min_wl
        self.max_wl = max_wl
        self.wavelengths = np.linspace(min_wl, max_wl, n_wl)

        # The data might have already been treated : it is faster to read the data from file
        dbfile = os.path.join(self.saving_dir, f"optical_polyanskiy_{self.min_wl}_{self.max_wl}_{self.n_wl}.csv")

        if os.path.isfile(dbfile):
            data = pd.read_csv(dbfile)
            data.set_index("Compound", inplace=True)
            self.data = data
        else:
            # Recompute the data file from the database
            warnings.warn(
                """Datafile not existing for these wavelengths: recollecting the data from the database.
                   This can take a few seconds..."""
            )
            self.data = self._get_optical_data_from_database()
            self.data.to_csv(dbfile)
            warnings.warn("The data has been collected and stored.")

        self.elem_data = self._get_element_props()

        # Split into bins
        bins *= len(props)
        slices = np.linspace(0, len(self.elem_data.T), bins + 1, True).astype(int)
        counts = np.diff(slices)

        cols = self.elem_data.columns[slices[:-1] + counts // 2]
        labels = list(cols)  # [col for col in cols]

        all_element_data = pd.DataFrame(
            np.add.reduceat(self.elem_data.values, slices[:-1], axis=1) / counts,
            columns=cols,
            index=self.elem_data.index,
        )

        self.all_element_data = all_element_data
        self.prop_names = labels

        self.impute_nan = impute_nan
        if self.impute_nan:
            self.all_element_data.fillna(self.all_element_data.mean(), inplace=True)
        else:
            warnings.warn(f"{self.__class__.__name__}(impute_nan=False):\n" + IMPUTE_NAN_WARNING)

    def _get_element_props(self):
        """
        Returns the properties of single elements from the data contained in the database.
        """

        data = self.data.copy()

        cols = []
        if "refractive" in self.props:
            cols += [name for name in data.columns if "n_" in name]
        if "extinction" in self.props:
            cols += [name for name in data.columns if "k_" in name]
        if "reflectivity" in self.props:
            cols += [name for name in data.columns if "R_" in name]

        data = data[cols]

        # Compute the exact values
        res = []
        elem, elem_absent = get_elem_in_data(data, as_pure=True)
        for e in elem:
            res.append(data.loc[e].values)
        for e in elem_absent:
            res.append(np.nan * np.ones(len(self.props) * self.n_wl))

        res = np.array(res)
        df_exact = pd.DataFrame(res, columns=cols, index=pd.Index(elem + elem_absent))

        if self.method == "exact":
            return df_exact
        else:
            # Compute the pseudo-inversed values
            df_pi = get_pseudo_inverse(self.data, cols)

            if self.method == "pseudo_inverse":
                return df_pi
            elif self.method == "combined":
                res_combined = []
                for i, e in df_exact.iterrows():
                    if e.isnull().sum() == 0:
                        res_combined.append(e.values)
                    else:
                        res_combined.append(df_pi.loc[i].values)
                res_combined = np.array(res_combined)
                df_combined = pd.DataFrame(res_combined, columns=cols, index=df_exact.index)
                return df_combined

            else:
                raise ValueError("The method should be either exact, pseudo_inverse or combined.")

    def _get_optical_data_from_database(self):
        """
        Get a dataframe with the refractive index, extinction coefficients, and reflectivity
        as obtained from the initial database, for an array of wavelengths.
        We need to handle the database that is in different formats...

        Returns:
            DataFrame with the data
        """

        db_dir = os.path.join(self.saving_dir, "database")
        # The database has been compressed, it needs to be untarred if it is not already the case.
        if not os.listdir(db_dir):
            db_file = os.path.join(module_dir, "data_files/optical_polyanskiy/database.tar.xz")
            with tarfile.open(db_file, mode="r:xz") as tar:
                tar.extractall(self.saving_dir)

        names = []
        compos = []
        N = []
        K = []

        for material in os.listdir(db_dir):
            # Some materials have the data in the needed wavelengths range,
            # others don't and throw an error
            try:
                n_avg = self._get_nk_avg(os.path.join(db_dir, material))
                compos.append(Composition(material))
                names.append(material)
                N.append(n_avg.real)
                K.append(n_avg.imag)
            except ValueError:
                pass

        W = 1000 * self.wavelengths
        N = np.array(N)
        K = np.array(K)
        R = ((N - 1) ** 2 + K**2) / ((N + 1) ** 2 + K**2)

        cols_names_n = [f"n_{np.round(wl, 2)}" for wl in W]
        cols_names_k = [f"k_{np.round(wl, 2)}" for wl in W]
        cols_names_r = [f"R_{np.round(wl, 2)}" for wl in W]

        df_n = pd.DataFrame(N, columns=cols_names_n)
        df_k = pd.DataFrame(K, columns=cols_names_k)
        df_r = pd.DataFrame(R, columns=cols_names_r)

        df = pd.concat([df_n, df_k, df_r], axis=1)
        df["Composition"] = compos
        df["Compound"] = names
        df.set_index("Compound", inplace=True)

        return df

    def _get_optical_data_from_file(self, yaml_file):
        """
        From a yml file, returns the refractive index for a given wavelength

        Args:
            yaml_file: path to the yml file containing the data

        Returns:
            refractive index (complex number)
        """

        # Open the yml file
        with open(yaml_file) as yml:
            data = yaml.YAML(typ="safe", pure=True).load(yml)["DATA"][0]

        data_format = data["type"]

        # We now treat different formats for the data
        if data_format in ["tabulated nk", "tabulated n"]:
            # We parse the data to get the wavelength, n and kappa
            if data_format == "tabulated nk":
                arr = np.fromstring(data["data"].replace("\n", " "), sep=" ").reshape((-1, 3))
                K = arr[:, 2]
            # kappa not available -> 0
            elif data_format == "tabulated n":
                arr = np.fromstring(data["data"].replace("\n", " "), sep=" ").reshape((-1, 2))
                K = np.zeros(len(arr))

            wl = arr[:, 0]
            range_wl = np.array([np.min(wl), np.max(wl)])
            N = arr[:, 1]

            interpN = interp1d(wl, N)
            interpK = interp1d(wl, K)

            # Check that self.wavelengths are within the range
            if np.any(self.min_wl < range_wl[0]) or np.any(self.max_wl > range_wl[1]):
                raise ValueError(
                    f"""The values of lambda asked to be returned is outside the range
                of available data. This can lead to strong deviation as extrapolation might be bad. For information, the
                range is [{range_wl[0]}, {range_wl[1]}] microns."""
                )
            else:
                return np.array([x for x in np.nditer(interpN(self.wavelengths) + 1j * interpK(self.wavelengths))])

        # If the data is not tabulated, it is given with a formula
        elif "formula" in data_format:
            range_wl = np.fromstring(data["wavelength_range"], sep=" ")
            # Check that lamb is within the range
            if np.any(self.min_wl < range_wl[0]) or np.any(self.max_wl > range_wl[1]):
                raise ValueError(
                    f""""The values of lambda asked to be returned is outside the range
                of available data. This can lead to strong deviation as extrapolation might be bad. For information, the
                range is [{range_wl[0]}, {range_wl[1]}] microns."""
                )
            else:
                coeff_file = np.fromstring(data["coefficients"], sep=" ")

                N = np.zeros(self.n_wl) + 1j * np.zeros(self.n_wl)

                if data_format == "formula 1":
                    coeffs = np.zeros(17)
                    coeffs[0 : len(coeff_file)] = coeff_file
                    N += 1 + coeffs[0]
                    for i in range(1, 17, 2):
                        N += (coeffs[i] * self.wavelengths**2) / (self.wavelengths**2 - coeffs[i + 1] ** 2)
                    N = np.sqrt(N)

                elif data_format == "formula 2":
                    coeffs = np.zeros(17)
                    coeffs[0 : len(coeff_file)] = coeff_file
                    N += 1 + coeffs[0]
                    for i in range(1, 17, 2):
                        N += (coeffs[i] * self.wavelengths**2) / (self.wavelengths**2 - coeffs[i + 1])
                    N = np.sqrt(N)

                elif data_format == "formula 3":
                    coeffs = np.zeros(17)
                    coeffs[0 : len(coeff_file)] = coeff_file
                    N += coeffs[0]
                    for i in range(1, 17, 2):
                        N += coeffs[i] * self.wavelengths ** coeffs[i + 1]
                    N = np.sqrt(N)

                elif data_format == "formula 4":
                    coeffs = np.zeros(17)
                    coeffs[0 : len(coeff_file)] = coeff_file
                    N += coeffs[0]
                    N += coeffs[1] * self.wavelengths ** coeffs[2] / (self.wavelengths**2 - coeffs[3] ** coeffs[4])
                    N += coeffs[5] * self.wavelengths ** coeffs[6] / (self.wavelengths**2 - coeffs[7] ** coeffs[8])
                    for i in range(9, 17, 2):
                        N += coeffs[i] * self.wavelengths ** coeffs[i + 1]
                    N = np.sqrt(N)

                elif data_format == "formula 5":
                    coeffs = np.zeros(11)
                    coeffs[0 : len(coeff_file)] = coeff_file
                    N += coeffs[0]
                    for i in range(1, 11, 2):
                        N += coeffs[i] * self.wavelengths ** coeffs[i + 1]

                elif data_format == "formula 6":
                    coeffs = np.zeros(11)
                    coeffs[0 : len(coeff_file)] = coeff_file
                    N += 1 + coeffs[0]
                    for i in range(1, 11, 2):
                        N += coeffs[i] * self.wavelengths**2 / (coeffs[i + 1] * self.wavelengths**2 - 1)

                elif data_format == "formula 7":
                    coeffs = np.zeros(6)
                    coeffs[0 : len(coeff_file)] = coeff_file
                    N += (
                        coeffs[0]
                        + coeffs[1] / (self.wavelengths**2 - 0.028)
                        + coeffs[2] / (self.wavelengths**2 - 0.028) ** 2
                    )
                    N += (
                        coeffs[3] * self.wavelengths**2
                        + coeffs[4] * self.wavelengths**4
                        + coeffs[5] * self.wavelengths**6
                    )

                elif data_format == "formula 8":
                    coeffs = np.zeros(4)
                    coeffs[0 : len(coeff_file)] = coeff_file
                    N += (
                        coeffs[0]
                        + coeffs[1] * self.wavelengths**2 / (self.wavelengths**2 - coeffs[2])
                        + coeffs[3] * self.wavelengths**2
                    )
                    N = np.sqrt((1 + 2 * N) / (1 - N))

                elif data_format == "formula 9":
                    coeffs = np.zeros(6)
                    coeffs[0 : len(coeff_file)] = coeff_file
                    N += (
                        coeffs[0]
                        + coeffs[1] / (self.wavelengths**2 - coeffs[2])
                        + coeffs[3] * (self.wavelengths - coeffs[4]) / ((self.wavelengths - coeffs[4]) ** 2 + coeffs[5])
                    )
                    N = np.sqrt(N)

                return N + 1j * np.zeros(len(N))

        else:
            raise ValueError("UnsupportedDataType: This data type is currently not supported !")

    def _get_nk_avg(self, dirname):
        """
        Compute the average of n and ĸ for a compound of the database.

        Args:
            dirname: path to the compound of interest

        Returns:
            refractive index (complex number)
        """

        files = os.listdir(dirname)

        navg = []
        for f in files:
            # Some files have only kappa : they raise a ValueError
            try:
                # We get the average of all curves in the region of interest.
                # For some systems, the region of interest is not covered.
                navg.append(self._get_optical_data_from_file(os.path.join(dirname, f)))
            except ValueError:
                pass

        # We go further only if there is data
        if navg:
            navg = np.array(navg).mean(axis=0)
            Navg = navg.real
            Kavg = navg.imag
            return Navg + 1j * Kavg
        else:
            raise ValueError(f"No correct data for {dirname}")

    def get_elemental_property(self, elem, property_name):
        estr = str(elem)
        return self.all_element_data.loc[estr][property_name]


class TransportData(AbstractData):
    """
    Class to use transport data from Ricci et al., see
    An ab initio electronic transport database for inorganic materials.
    Ricci, F., Chen, W., Aydemir, U., Snyder, G. J., Rignanese, G. M., Jain, A., & Hautier, G. (2017).
    Scientific data, 4(1), 1-13.
    https://doi.org/10.1038/sdata.2017.85

    The database has been used to extract:
    1) the properties of single elements when available.
       These are stored in matminer/utils/data_files/mp_transport/transport_pure_elems.csv
    2) the pseudo-inverse of the properties of single elements.
       These pseudo-inverse contributions correspond to the coefficients of a least-square fit
       from the compositions to the properties. This can allow to better take into account
       data from different compounds for a given element.

    Using the pseudo-inverses (method="pseudo_inverse") instead of
    the elemental properties (method="exact") leads to better results as far as we have checked.
    Another possibility is to use method="combined", where the exact values are taken
    for compounds present as pure compounds in the database, and the pseudo-inverse
    is taken if the element is not present purely in the database.

    For the effective mass, the pseudo-inverse is obtained on 1/(alpha+m),
    then m is re-obtained for single elements. This is to avoid
    huge errors coming from the huge spread in data (12 orders of magnitude).

    Args:
        props: optical properties to include. Should be a (sub)list of
               ["sigma_p", "sigma_n", "S_p", "S_n", "kappa_p", "kappa_n", "PF_p", "PF_n", "m_p", "m_n"]
               for the hole (_p) and electron (_n) conductivity (sigma), Seebeck coefficient (S),
               thermal conductivity (kappa), power factor (PF) and effective mass (m).
        method: type of values, either "exact", "pseudo_inverse", or "combined".
        alpha: Value used to featurize the effective mass.
               The values of the effective masses span 12 orders of magnitude, which makes the pseudo-inverse biased
               To overcome this, we use 1 / (alpha + m) for the pseudo-inversion.
               The value of alpha can be tested. A file for each of them is created,
               so that it is not computed each time.
               Defaults to 0, and used only if method != "exact".
        saving_dir: folder to save the data and csv file used for the featurization. Saving them helps fasten the
                    featurization.
        impute_nan (bool): if True, the features for the elements
            that are missing from the data_source or are NaNs are replaced by the
            average of each features over the available elements.
    """

    def __init__(
        self,
        props=None,
        method="pseudo_inverse",
        alpha=0,
        saving_dir="~/.matminer/transport_props/",
        impute_nan=False,
    ):
        # Handles the saving folder
        saving_dir = os.path.expanduser(saving_dir)
        os.makedirs(saving_dir, exist_ok=True)
        self.saving_dir = saving_dir

        # Handle the selection of properties
        possible_props = ["sigma_p", "sigma_n", "S_p", "S_n", "kappa_p", "kappa_n", "PF_p", "PF_n", "m_p", "m_n"]
        if props is None:
            props = possible_props
        elif not all([prop in possible_props for prop in props]):
            raise ValueError("This property is not available: choose from " + ", ".join(possible_props))

        self.props = props
        self.method = method
        self.alpha = alpha

        # Read the database as a compressed or csv file
        dfile = os.path.join(module_dir, "data_files/mp_transport/", "transport_database.csv")
        dfile_tarred = os.path.join(module_dir, "data_files/mp_transport/", "transport_database.tar.xz")
        if not os.path.isfile(dfile) and os.path.isfile(dfile_tarred):
            with tarfile.open(dfile_tarred, mode="r:xz") as tar:
                tar.extractall(os.path.join(module_dir, "data_files/mp_transport/"))
        data = pd.read_csv(dfile).drop(columns=["mp_id"])
        data.rename(columns={"pretty_formula": "Compound"}, inplace=True)
        data.set_index("Compound", inplace=True)

        # Remove outliers
        data = data[(np.abs(stats.zscore(data)) < 3).all(axis=1)]

        # Add composition
        compos = []
        for c in data.index:
            compos.append(Composition(c))
        data["Composition"] = compos

        self.data = data

        self.all_element_data = self._get_element_props()

        self.prop_names = list(self.all_element_data.columns)

        self.impute_nan = impute_nan
        if self.impute_nan:
            self.all_element_data.fillna(self.all_element_data.mean(), inplace=True)
        else:
            warnings.warn(f"{self.__class__.__name__}(impute_nan=False):\n" + IMPUTE_NAN_WARNING)

    def _get_element_props(self):
        #
        # Compute the exact values
        #
        dfile = os.path.join(module_dir, "data_files/mp_transport/", "transport_pure_elems.csv")
        df_exact = pd.read_csv(dfile)
        df_exact.set_index("Element", inplace=True)
        df_exact.drop(columns=["mp_id"], inplace=True)

        elem, elem_absent = get_elem_in_data(df_exact, as_pure=True)
        res = []
        for e in elem:
            res.append(df_exact.loc[e][self.props].values)
        for e in elem_absent:
            res.append(np.nan * np.ones(len(self.props)))

        res = np.array(res)
        cols = self.props
        df_exact = pd.DataFrame(res, columns=cols, index=elem + elem_absent)
        df_exact = df_exact[self.props]

        if self.method == "exact":
            return df_exact
        else:
            #
            # Retrieve or compute the pseudo-inversed values.
            #
            dbfile = os.path.join(self.saving_dir, f"transport_pi_{self.alpha}.csv")
            if os.path.isfile(dbfile):
                df_pi = pd.read_csv(dbfile)
                df_pi.set_index("Element", inplace=True)
                df_pi = df_pi[self.props]
            else:
                warnings.warn(
                    """Pseudo-inverse values not found for this value of alpha. Recomputing them...
                       This can take a few seconds..."""
                )
                TP = self.data.copy()
                TP["1/m_p"] = 1 / (self.alpha + TP["m_p"].values)
                TP["1/m_n"] = 1 / (self.alpha + TP["m_n"].values)

                df_pi = get_pseudo_inverse(TP)

                df_pi["m_p"] = 1 / df_pi["1/m_p"].values - self.alpha
                df_pi["m_n"] = 1 / df_pi["1/m_n"].values - self.alpha
                df_pi.drop(columns=["1/m_p", "1/m_n"], inplace=True)
                df_pi.index.name = "Element"

                df_pi.to_csv(dbfile)
                warnings.warn("The pseudo-inverse coefficients have been collected and stored.")

            if self.method == "pseudo_inverse":
                return df_pi
            elif self.method == "combined":
                res_combined = []
                for i, e in df_exact.iterrows():
                    if e.isnull().sum() == 0:
                        res_combined.append(e.values)
                    else:
                        res_combined.append(df_pi.loc[i].values)
                res_combined = np.array(res_combined)
                df_combined = pd.DataFrame(res_combined, columns=df_exact.columns, index=df_exact.index)
                return df_combined
            else:
                raise ValueError("The method should be either exact or pseudo_inverse.")

    def get_elemental_property(self, elem, property_name):
        estr = str(elem)
        return self.all_element_data.loc[estr][property_name]
