"""
Utility classes for retrieving elemental properties. Provides
a uniform interface to several different elemental property resources
including ``pymatgen`` and ``Magpie``.
"""

import abc
import json
import os
from glob import glob

import numpy as np
import pandas as pd
from pymatgen.core.periodic_table import Element, _pt_data

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
        with open(os.path.join(module_dir, "data_files", "cohesive_energies.json")) as f:
            self.cohesive_energy_data = json.load(f)

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
    """

    def __init__(self):
        from matminer.utils.data_files.deml_elementdata import properties

        self.all_props = properties
        self.available_props = list(self.all_props.keys()) + [
            "formal_charge",
            "valence_s",
            "valence_p",
            "valence_d",
            "first_ioniz",
            "total_ioniz",
        ]

        # Compute the FERE correction energy
        fere_corr = {}
        for k, v in self.all_props["GGAU_Etot"].items():
            fere_corr[k] = self.all_props["mus_fere"][k] - v
        self.all_props["FERE correction"] = fere_corr

        # List out the available charge-dependent properties
        self.charge_dependent_properties = [
            "xtal_field_split",
            "magn_moment",
            "so_coupling",
            "sat_magn",
        ]

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


    Finding the exact meaning of each of these features can be quite difficult.
    Reproduced in ./data_files/magpie_elementdata_feature_descriptions.txt.
    """

    def __init__(self):
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
                for atomic_no in range(1, len(_pt_data) + 1):  # max Z=103
                    try:
                        if descriptor_name in ["OxidationStates"]:
                            prop_value = [float(i) for i in lines[atomic_no - 1].split()]
                        else:
                            prop_value = float(lines[atomic_no - 1])
                    except (ValueError, IndexError):
                        prop_value = float("NaN")
                    self.all_elemental_props[descriptor_name][Element.from_Z(atomic_no).symbol] = prop_value

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
        return elem.common_oxidation_states if self.use_common_oxi_states else elem.oxidation_states

    def get_charge_dependent_property(self, element, charge, property_name):
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
    """

    def __init__(self):
        mixing_dataset = pd.read_csv(
            os.path.join(module_dir, "data_files", "MiedemaLiquidDeltaHf.tsv"),
            delim_whitespace=True,
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
    """

    def __init__(self):
        dfile = os.path.join(module_dir, "data_files/matscholar_els.json")
        with open(dfile) as fp:
            embeddings = json.load(fp)
        self.prop_names = [f"embedding {i}" for i in range(1, 201)]
        all_element_data = {}
        for el, embedding in embeddings.items():
            all_element_data[el] = dict(zip(self.prop_names, embedding))
        self.all_element_data = all_element_data

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
    """

    def __init__(self):
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
        self.params = pd.read_csv(
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
        new_params = self.params.append(new_data, sort=True, ignore_index=True)
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
