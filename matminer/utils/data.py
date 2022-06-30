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
from pymatgen.core import Composition
from numpy.linalg import pinv
import yaml
from scipy.interpolate import interp1d

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


class OpticalData(AbstractData):
    """
    Class to use optical data from https://www.refractiveindex.info
    The properties are the refractive index n and the extinction coefficient ĸ
    (measured or computed with DFT), and the reflectivity R as obtained from
    Fresnel's equation.
    Data has been initially considered if available from 380 to 780 nm,
    but other ranges can be chosen as well.

    The initial database has been used to extract:
    1) the properties of single elements when available.
    2) the pseudo-inverse of the properties of single elements,
       based on the data for ~200 compounds.
    Using the pseudo-inverses instead of the elemental properties
    leads to better results as far as we have checked.

    n, ĸ, and R are spectra. We split these spectra into bins (initially 10)
    where their average values are taken as features.

    Args:
        bins: number of bins to split the spectra. This is also the number
              of elemental features for each property.
        props: optical properties to include. Should be a list with
               'refractive' and/or 'extinction' and/or 'reflectivity.
        pseudo_inverse: whether to use or not the pseudo-inversed values
        method: type of values, either 'exact', 'pseudo_inverse', or 'combined'
                if 'combined', takes the exact values when available, and the pseudo-inversed
                ones otherwise
        min_wl: minimum wavelength to include in the spectra (µm) - before binning
        max_wl : maximum wavelength to include in the spectra (µm)
        n_wl: number of wavelengths to include in the spectra
    """

    def __init__(self, bins=10, props=None, method='combined',
                 min_wl=0.38, max_wl=0.78, n_wl=401):

        # Handle the selection of properties
        if props is None:
            props = ['refractive', 'extinction', 'reflectivity']
        else:
            if not all([prop in ['refractive', 'extinction', 'reflectivity'] for prop in props]):
                raise ValueError('This property is not available: choose from refractive, extinction, or reflectivity')

        self.props = props
        self.method = method
        self.n_wl = n_wl
        self.min_wl = min_wl
        self.max_wl = max_wl
        self.wavelengths = np.linspace(min_wl, max_wl, n_wl)

        # The data has already been treated for the default values : this is faster in this case
        if self.min_wl == 0.38 and self.max_wl == 0.78 and self.n_wl == 401:
            dfile = os.path.join(module_dir, "data_files/optical_polyanskiy/optical_polyanskiy.csv")
            data = pd.read_csv(dfile)
            data.set_index('Compound', inplace=True)
            self.data = data
        else:
            # Recompute the data file from the database
            print("Selecting non-default wavelengths: recollecting the data from the database...")
            self.data = self._get_optical_data_from_database()
            print("The data has been collected.")

        self.elem_data = self._get_element_props()

        # Split into bins
        bins *= len(props)
        slices = np.linspace(0, len(self.elem_data.T), bins + 1, True).astype(int)
        counts = np.diff(slices)

        cols = self.elem_data.columns[slices[:-1] + counts // 2]
        labels = [col for col in cols]

        all_element_data = pd.DataFrame(np.add.reduceat(self.elem_data.values, slices[:-1], axis=1) / counts,
                                        columns=cols, index=self.elem_data.index)

        self.all_element_data = all_element_data
        self.prop_names = labels

    def _get_elem_in_data(self, as_pure=False):
        """
        Look for all elements present in the compounds from the dataframe

        Args:
             as_pure: if True, consider only the pure compounds

        Returns:
            List of elements (str)
        """
        elements = ['H', 'He',
                    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
                    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co',
                    'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
                    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh',
                    'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
                    'Cs', 'Ba', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt',
                    'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
                    'Fr', 'Ra', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
                    'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og',
                    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb',
                    'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Ac', 'Th', 'Pa',
                    'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']

        elems_in_df = []
        elems_not_in_df = []

        df = self.data.copy()

        # Find the elements in the data, as pure or not
        if not as_pure:
            for elem in elements:
                for compound in df.index.to_list():
                    if elem in compound and elem not in elems_in_df:
                        elems_in_df.append(elem)

        else:
            for elem in elements:
                if elem in df.index:
                    elems_in_df.append(elem)

        # Find the elements not in the data
        for elem in elements:
            if elem not in elems_in_df:
                elems_not_in_df.append(elem)

        return elems_in_df, elems_not_in_df

    def _get_element_props(self):
        """
        Returns the properties of single elements from the data contained in the database.
        """

        data = self.data.copy()

        cols = []
        if 'refractive' in self.props:
            cols += [name for name in data.columns if 'n_' in name]
        if 'extinction' in self.props:
            cols += [name for name in data.columns if 'k_' in name]
        if 'reflectivity' in self.props:
            cols += [name for name in data.columns if 'R_' in name]

        data = data[cols]

        # Compute the exact values
        res = []
        elem, elem_absent = self._get_elem_in_data(as_pure=True)
        for e in elem:
            res.append(data.loc[e].values)
        for e in elem_absent:
            res.append(np.nan * np.ones(len(self.props) * self.n_wl))

        res = np.array(res)
        df_exact = pd.DataFrame(res, columns=cols, index=pd.Index(elem + elem_absent))

        # Compute the pseudo-inversed values
        # We have to create a matrix with n_elem_tot (all available element in the database) cols and len(df) rows,
        # containing the compositions of each material from the database
        elems_in_df, elems_not_in_df = self._get_elem_in_data(as_pure=False)
        n_elem_tot = len(elems_in_df)

        # Initialize the matrix
        A = np.zeros([len(self.data), n_elem_tot])

        compos = self.data['Composition']
        for i, comp in enumerate(compos):
            comp = Composition(comp)
            for j in range(n_elem_tot):
                if elems_in_df[j] in comp:
                    A[i, j] = comp.get_atomic_fraction(elems_in_df[j])

        pi_A = pinv(A)

        res_pi = pi_A @ data.values
        res_pi = np.vstack([res_pi, np.nan * np.ones([len(elems_not_in_df), len(self.props) * self.n_wl])])
        df_pi = pd.DataFrame(res_pi, columns=cols, index=pd.Index(elems_in_df + elems_not_in_df))

        if self.method == 'exact':
            return df_exact
        elif self.method == 'pseudo_inverse':
            return df_pi
        elif self.method == 'combined':
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
            raise ValueError('The method should be either exact or pseudo_inverse.')

    def _get_optical_data_from_database(self):
        """
        Get a dataframe with the refractive index, extinction coefficients, and reflectivity
        as obtained from the initial database, for an array of wavelengths.
        We need to handle the database that is in different formats...

        Returns:
            DataFrame with the data
        """

        db_dir = [os.path.join(module_dir, "data_files/optical_polyanskiy/database/other/semiconductor alloys/"),
                  os.path.join(module_dir, 'data_files/optical_polyanskiy/database/other/alloys'),
                  os.path.join(module_dir, 'data_files/optical_polyanskiy/database/main'),
                  os.path.join(module_dir, 'data_files/optical_polyanskiy/database/other/intermetallics')]

        names = []
        compos = []
        N = []
        K = []

        for dir in db_dir:
            for material in os.listdir(dir):
                # Some materials have the data in the needed wavelengths range,
                # others don't and throw an error
                try:
                    if 'main' in dir:
                        n_avg = self._get_nk_avg(os.path.join(dir, material))
                        compos.append(Composition(material))
                        names.append(material)
                        N.append(n_avg.real)
                        K.append(n_avg.imag)
                    elif 'intermetallics' in dir:
                        n_avg = self._get_nk_avg(os.path.join(dir, material))
                        names.append(material)
                        compos.append(Composition(material))
                        N.append(n_avg.real)
                        K.append(n_avg.imag)
                    elif 'alloys' in dir:
                        files = os.listdir(dir + '/' + material)
                        alloy = ''
                        for f in files:
                            if material == 'Au-Ag':
                                alloy = f[6:-4]
                            elif material == 'Cu-Zn':
                                alloy = f[7:-4]
                            elif material == 'Ni-Fe':
                                alloy = 'Ni80Fe20'
                            elif material == 'AlAs-GaAs':
                                x_Al = float(f.split('-')[1][:-4])
                                alloy = 'Al' + str(x_Al) + 'Ga' + str(np.round(100 - x_Al, 2)) + 'As100'
                            elif material == 'AlN-Al2O3':
                                x_N = np.round(float(f.split('-')[1][:-4]), 2)
                                x_Al = np.round((64 + x_N) / 3, 2)
                                x_O = np.round(32 - x_N, 2)
                                alloy = 'Al' + str(x_Al) + 'O' + str(x_O) + 'N' + str(x_N)
                            elif material == 'AlSb-GaSb':
                                x_Al = np.round(float(f.split('-')[1][:-4]), 2)
                                x_Sb = 100
                                x_Ga = np.round(100 - x_Al, 2)
                                alloy = 'Al' + str(x_Al) + 'Ga' + str(x_Ga) + 'Sb' + str(x_Sb)
                            elif material == 'GaAs-InAs':
                                alloy = 'In52Ga48As100'
                            elif material == 'GaAs-InAs-GaP-InP':
                                alloy = 'In52Ga48As24P76'
                            elif material == 'GaP-InP':
                                alloy = 'Ga51In49P100'
                            elif material == 'Si-Ge':
                                x_Si = np.round(float(f.split('-')[1][:-4]), 2)
                                x_Ge = np.round(100 - x_Si, 2)
                                alloy = 'Si' + str(x_Si) + 'Ge' + str(x_Ge)
                            else:
                                raise Warning('Material of unknown type: ' + material)

                            n_avg = self._get_optical_data_from_file(os.path.join(dir + '/' + material + '/' + f))

                            names.append(alloy)
                            compos.append(Composition(alloy))
                            N.append(n_avg.real)
                            K.append(n_avg.imag)

                except ValueError:
                    pass

        W = 1000 * self.wavelengths
        N = np.array(N)
        K = np.array(K)
        R = ((N - 1) ** 2 + K ** 2) / ((N + 1) ** 2 + K ** 2)

        cols_names_n = ['n_{0}'.format(np.round(wl, 2)) for wl in W]
        cols_names_k = ['k_{0}'.format(np.round(wl, 2)) for wl in W]
        cols_names_r = ['R_{0}'.format(np.round(wl, 2)) for wl in W]

        df_n = pd.DataFrame(N, columns=cols_names_n)
        df_k = pd.DataFrame(K, columns=cols_names_k)
        df_r = pd.DataFrame(R, columns=cols_names_r)

        df = pd.concat([df_n, df_k, df_r], axis=1)
        df['Composition'] = compos
        df['Compound'] = names
        df.set_index('Compound', inplace=True)

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
        with open(yaml_file, 'r') as yml:
            data = yaml.safe_load(yml)['DATA'][0]

        data_format = data['type']

        # We now treat different formats for the data
        if data_format in ['tabulated nk', 'tabulated n']:
            # We parse the data to get the wavelength, n and kappa
            if data_format == 'tabulated nk':
                arr = np.fromstring(data['data'].replace('\n', ' '), sep=' ').reshape((-1, 3))
                K = arr[:, 2]
            # kappa not available -> 0
            elif data_format == 'tabulated n':
                arr = np.fromstring(data['data'].replace('\n', ' '), sep=' ').reshape((-1, 2))
                K = np.zeros(len(arr))

            wl = arr[:, 0]
            range_wl = np.array([np.min(wl), np.max(wl)])
            N = arr[:, 1]

            interpN = interp1d(wl, N)
            interpK = interp1d(wl, K)

            # Check that self.wavelengths are within the range
            if np.any(self.min_wl < range_wl[0]) or np.any(self.max_wl > range_wl[1]):
                raise ValueError(""""The values of lambda asked to be returned is outside the range 
                of available data. This can lead to strong deviation as extrapolation might be bad. For information, the
                range is [{0}, {1}] microns.""".format(range_wl[0], range_wl[1]))
            else:
                return np.array([x for x in np.nditer(interpN(self.wavelengths) + 1j * interpK(self.wavelengths))])

        # If the data is not tabulated, it is given with a formula
        elif 'formula' in data_format:
            range_wl = np.fromstring(data['wavelength_range'], sep=' ')
            # Check that lamb is within the range
            if np.any(self.min_wl < range_wl[0]) or np.any(self.max_wl > range_wl[1]):
                raise ValueError(""""The values of lambda asked to be returned is outside the range 
                of available data. This can lead to strong deviation as extrapolation might be bad. For information, the
                range is [{0}, {1}] microns.""".format(range_wl[0], range_wl[1]))
            else:
                coeff_file = np.fromstring(data['coefficients'], sep=' ')

                N = np.zeros(self.n_wl) + 1j * np.zeros(self.n_wl)

                if data_format == 'formula 1':
                    coeffs = np.zeros(17)
                    coeffs[0:len(coeff_file)] = coeff_file
                    N += 1 + coeffs[0]
                    for i in range(1, 17, 2):
                        N += ((coeffs[i] * self.wavelengths ** 2) / (self.wavelengths ** 2 - coeffs[i + 1] ** 2))
                    N = np.sqrt(N)

                elif data_format == 'formula 2':
                    coeffs = np.zeros(17)
                    coeffs[0:len(coeff_file)] = coeff_file
                    N += 1 + coeffs[0]
                    for i in range(1, 17, 2):
                        N += ((coeffs[i] * self.wavelengths ** 2) / (self.wavelengths ** 2 - coeffs[i + 1]))
                    N = np.sqrt(N)

                elif data_format == 'formula 3':
                    coeffs = np.zeros(17)
                    coeffs[0:len(coeff_file)] = coeff_file
                    N += coeffs[0]
                    for i in range(1, 17, 2):
                        N += coeffs[i] * self.wavelengths ** coeffs[i + 1]
                    N = np.sqrt(N)

                elif data_format == 'formula 4':
                    coeffs = np.zeros(17)
                    coeffs[0:len(coeff_file)] = coeff_file
                    N += coeffs[0]
                    N += coeffs[1] * self.wavelengths ** coeffs[2] / (self.wavelengths ** 2 - coeffs[3] ** coeffs[4])
                    N += coeffs[5] * self.wavelengths ** coeffs[6] / (self.wavelengths ** 2 - coeffs[7] ** coeffs[8])
                    for i in range(9, 17, 2):
                        N += coeffs[i] * self.wavelengths ** coeffs[i + 1]
                    N = np.sqrt(N)

                elif data_format == 'formula 5':
                    coeffs = np.zeros(11)
                    coeffs[0:len(coeff_file)] = coeff_file
                    N += coeffs[0]
                    for i in range(1, 11, 2):
                        N += coeffs[i] * self.wavelengths ** coeffs[i + 1]

                elif data_format == 'formula 6':
                    coeffs = np.zeros(11)
                    coeffs[0:len(coeff_file)] = coeff_file
                    N += 1 + coeffs[0]
                    for i in range(1, 11, 2):
                        N += coeffs[i] * self.wavelengths ** 2 / (coeffs[i + 1] * self.wavelengths ** 2 - 1)

                elif data_format == 'formula 7':
                    coeffs = np.zeros(6)
                    coeffs[0:len(coeff_file)] = coeff_file
                    N += coeffs[0] + coeffs[1] / (self.wavelengths ** 2 - 0.028) + coeffs[2] / \
                         (self.wavelengths ** 2 - 0.028) ** 2
                    N += coeffs[3] * self.wavelengths ** 2 + coeffs[4] * self.wavelengths ** 4 + \
                         coeffs[5] * self.wavelengths ** 6

                elif data_format == 'formula 8':
                    coeffs = np.zeros(4)
                    coeffs[0:len(coeff_file)] = coeff_file
                    N += coeffs[0] + coeffs[1] * self.wavelengths ** 2 / (self.wavelengths ** 2 - coeffs[2]) + \
                         coeffs[3] * self.wavelengths ** 2
                    N = np.sqrt((1 + 2 * N) / (1 - N))

                elif data_format == 'formula 9':
                    coeffs = np.zeros(6)
                    coeffs[0:len(coeff_file)] = coeff_file
                    N += coeffs[0] + coeffs[1] / (self.wavelengths ** 2 - coeffs[2]) + \
                         coeffs[3] * (self.wavelengths - coeffs[4]) / ((self.wavelengths - coeffs[4]) ** 2 + coeffs[5])
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
            raise ValueError("No correct data for" + dirname)

    def get_elemental_property(self, elem, property_name):
        estr = str(elem)
        return self.all_element_data.loc[estr][property_name]
