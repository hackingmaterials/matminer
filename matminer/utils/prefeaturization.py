"""
Utilities for examining dataframe inputs before featurizing.

Reasons for doing this:
  * Featurizers may be out of scope for some inputs.
  * Featurizers may be prone to errors for many samples based on input
  * Featurizers may take a very long time depending on inputs
"""


"""Derive a list of meta-features of a given dataset to get recommendation of
featurizers.

The meta-features serve as indicators of the dataset characteristics that may
affect the choice of featurizers.

Based on the meta-features and then benchmarking (i) featurizer availability, 
(ii) featurizer importance and (iii) featurizer computational budget of 
existing featurizers on a variety of datasets, we can get a sense of how these
featurizers perform for datasets with different meta-features, and then make 
some strategies of featurizer selection.

When given a new dataset, we can compute its meta-features, and then get the 
recommended featurizers based on the pre-defined strategies (e.g. one way is 
to get the L1 distances of meta-features of all pre-defined strategies and 
meta-features of the new dataset, then find some "nearest" strategies and make
an estimation of computational budget, and finally taking all these factors 
together to make a final recommendation of featurizers)

Current meta-feat ures to be considered (many can be further added):
(i) Composition-related:
    - Number of compositions:
    - Percent of all-metallic alloys:
    - Percent of metallic-nonmetallic compounds:
    - Percent of nonmetallic compounds:
    - Number of elements present in the entire dataset: 
        e.g. can help to decided whether to use ChemicalSRO or Bob featurizers
        that can return O(N^2) features (increase rapidly with the number of 
        elements present)
    - Avg. number of elements in compositions:
    - Max. number of elements in compositions:
    - Min. number of elements in compositions:
    To do:
    - Percent of transitional-metal-containing alloys (dependency: percent of 
        all-metallic alloys): 
        e.g. can be used to determisne whether to use featurizers such as Miedema 
        that is more applicable to transitional alloys.
    - Percent of transitional-nonmetallic compounds (dependency: percent of 
        metallic-nonmetallic compounds): 
    - Prototypes of phases in the dataset:
        e.g. AB; AB2O4; MAX phase; etc maybe useful.
    - Percent of organic/molecules: 
        may need to call other packages e.g. deepchem or just fail this task as 
        we cannot directly support it in matminer.

(ii) Structure-related:
    - Percent of  ordered structures:
        e.g. can help to decide whether to use some featurizers that only apply
        to ordered structure such as GlobalSymmetryFeatures
    - Avg. number of atoms in structures:
        e.g. can be important for deciding on some extremely computational 
        expensive featurizers such as Voronoi-based ones or site statistics 
        ones such as SiteStatsFingerprint. They are assumed to be quite slow 
        if there are too many atoms in the structures.
    - Max. number of sites in structures: 
    To do:
    - Percent of 3D structure:
    - Percent of 2D structure:
    - Percent of 1D structure:

(iii) Missing_values-related:
    - Number of instances with missing_values
    - Percent of instances with missing_values
    - Number of missing_values
    - Percent of missing_values

To do:
(iv) Task-related:
    - Regression or Classification: 
        maybe some featurizers work better for classification or better for 
        regression
"""
import sys
import warnings
from automatminer.featurization.metaselection.metafeatures import \
    composition_mfs_dict, structure_mfs_dict
from collections import OrderedDict

from pymatgen.core.composition import Composition, Element
from pymatgen.core.structure import Structure, IStructure
from pymatgen.core.periodic_table import DummySpecie

import numpy as np
import pandas as pd
from functools import lru_cache
from pymatgen.core.composition import Composition
from pymatgen.core.structure import Structure, IStructure
from abc import ABCMeta, abstractmethod
from automatminer.featurization.metaselection.utils import \
    composition_statistics, \
    structure_statistics

__author__ = ["Qi Wang <wqthu11@gmail.com>", "Alex Dunn <ardunn@lbl.gov"]


def composition_statistics(compositions):
    """
    Get statistics of compositions. This is a helper function to the design
    of composition-related metafeatures.
    Args:
        compositions: a composition (str) or an iterable of compositions (list,
                      tuple, numpy.array or pandas.Series).
    """

    if isinstance(compositions, six.string_types):
        compositions = [compositions]
    stats = OrderedDict()
    for idx, composition in enumerate(compositions):
        stats[idx] = _composition_summary(composition)
    return stats


def _composition_summary(composition):
    """
    Extract some categorical messages from the composition.
    Args:
        composition (str): a given composition.
    Returns:
        dict of the following messages:
        major_category (int):
            all_metal: 1
            metal_nonmetal: 2
            all_nonmetal: 3, equal to organic? No
            unknown: 0
            ...to be continued

        minor_category (int):
            all_transitional_metal(except for rare_earth_metal/actinoid): 1
            all_rare_earth_metal: 2
            all_actinoid: 3
            all_alkali: 4
            all_alkaline: 5
            all_groupIIIA: 6
            unknown: 0
            ...to be continued

        prototype (int):
            double_perovskites: 1
            unknown: 0
            ...to be continued

        el_types_reduced ([int])：
            list of the irreducible categories of elements present in
            the composition, based on the predefined element_category function,
            sorted alphabetically.
            e.g. (1, 9) means there are both transitional metal and nonmetal
            in the composition.

        n_types (int):
            number of irreducible categories of elements in the composition.
            equal to the len(el_types_reduced).

        el_types ([int]):
            list of the unreduced categories of elements present in
            the composition, based on the predefined element_category function,
            sorted alphabetically.
            e.g. (1, 1, 9) means there are two types of transitional metal
            and one nonmetal in the composition.

        n_elements (int):
            number of elements in the composition.

        elements([str]):
            list of the symbols of elements in the composition.

    """
    try:
        c = Composition(composition)
    except BaseException:
        return {"major_composition_category": np.nan,
                "minor_composition_category": np.nan,
                "prototype": np.nan,
                "el_types_reduced": np.nan,
                "n_types": np.nan,
                "el_types": np.nan,
                "n_elements": np.nan,
                "elements": np.nan}
    elements = [x.symbol for x in c.elements]
    n_elements = len(c.elements)
    el_types = sorted([_element_category(x) for x in c.elements])
    n_types = len(el_types)
    el_types_reduced = list(set(el_types))

    major_category, minor_category = 0, 0
    # if there are only elements of one type, can be 1-11
    if len(el_types_reduced) == 1:
        if el_types_reduced[0] < 7:
            major_category = 1
        else:
            major_category = 3
        minor_category = el_types_reduced
    # if there are many types of metallic elements
    elif all([el_type < 7 for el_type in el_types_reduced]):
        major_category = 1
        minor_category = el_types_reduced  # just return the list for now
    elif any([el_type < 7 for el_type in el_types_reduced]):
        major_category = 2
        minor_category = el_types_reduced  # just return the list for now
    elif all([7 <= el_type < 11 for el_type in el_types_reduced]):
        major_category = 3

    prototype = _composition_prototype(composition)

    return {"major_composition_category": major_category,
            "minor_composition_category": minor_category,
            "prototype": prototype,
            "el_types_reduced": el_types_reduced,
            "n_types": n_types,
            "el_types": el_types,
            "n_elements": n_elements,
            "elements": elements}


def _composition_prototype(composition):
    """
    Guess the phase prototype from the integer anonymized_composition.
    Args:
        composition (str): a given composition.
    Returns:
        prototype:
            double_perovskites: 1
            unknown: 0
            ...to be continued

    """
    c = Composition(composition)
    c_int = Composition(c.get_integer_formula_and_factor()[0])
    f_int_anynomous = c_int.anonymized_formula
    prototype = 0
    if f_int_anynomous is "ABCDE6" and Element("O") in Composition(
            composition).elements:
        prototype = 1
    # to be continued
    return prototype


def _element_category(element):
    """
    Define the category of a given element.
    Args:
        element: an element symbol or a Pymatgen Element object
    Returns:
        metallic:
            is_transitional_metal(except for rare_earth_metal/actinoid): 1
            is_rare_earth_metal: 2
            is_actinoid: 3
            is_alkali: 4
            is_alkaline: 5
            is_groupIIIA_VIIA: 6 ("Al", "Ga", "In", "Tl", "Sn", "Pb",
                                  "Bi", "Po")
        non-metallic:
            is_metalloid: 7 ("B", "Si", "Ge", "As", "Sb", "Te", "Po")
            is_halogen: 8
            is_nonmetal: 9 ("C", "H", "N", "P", "O", "S", "Se")
            is_noble_gas: 10

        other-radiactive-etc:
            other: 11 (only a few elements are not covered by the
                       above categories)
    """
    if not isinstance(element, Element) and isinstance(element,
                                                       six.string_types):
        element = Element(element)
    if isinstance(element, DummySpecie):
        return 11
    elif element.is_transition_metal:
        if element.is_lanthanoid or element.symbol in {"Y", "Sc"}:
            return 2
        elif element.is_actinoid:
            return 3
        else:
            return 1
    elif element.is_alkali:
        return 4
    elif element.is_alkaline:
        return 5
    elif element.symbol in {"Al", "Ga", "In", "Tl", "Sn", "Pb", "Bi", "Po"}:
        return 6
    elif element.is_metalloid:
        return 7
    # elif element.is_chalcogen:
    #     return 8
    elif element.is_halogen:
        return 8
    elif element.symbol in {"C", "H", "N", "P", "O", "S", "Se"}:
        return 9
    elif element.is_noble_gas:
        return 10
    else:
        return 11


def structure_statistics(structures):
    """
    Get statistics of structures. This is a helper function to the design
    of strcture-related metafeatures.
    Args:
        structures: a Pymatgen Structure object or an iterable of Pymatgen
                    Structure objects (list, tuple, numpy.array or
                    pandas.Series).
    """
    if isinstance(structures, (Structure, IStructure)):
        structures = [structures]
    stats = OrderedDict()
    for idx, structure in enumerate(structures):
        stats[idx] = _structure_summary(structure)
    return stats


def _structure_summary(structure):
    """
    Extract messages from the structure.
    Args:
        structure: a Pymatgen Structure object
    Returns:
        dict of the following messages:
        nsites (int): number of sites in the structure.
        is_ordered (bool): whether the structure is ordered or not.
        ...to be continued

    """
    return {"n_sites": len(structure.sites),
            "is_ordered": structure.is_ordered}

class AbstractMetaFeature(object):
    """
    Abstract class for metafeature.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def calc(self, X, y):
        pass


class MetaFeature(AbstractMetaFeature):
    """
    Metafeature class.
    The dependence can be a metafeature class for a helper class.
    """

    def __init__(self, dependence=None):
        self.dependence = dependence
        super(MetaFeature, self).__init__()


class Helper(AbstractMetaFeature):
    """
    Helper class.
    """

    def __init__(self):
        super(Helper, self).__init__()


# composition-related metafeatures
def composition_stats(X):
    """
    Transform the input X to immutable tuple to use caching and call
    _composition_stats .
    Args:
        X: iterable compositions, can be Pymatgen Composition objects or
            string formulas

    Returns:
        _composition_stats
    """
    if isinstance(X, (pd.Series, pd.DataFrame)):
        return _composition_stats(tuple(X.values))
    if isinstance(X, (list, np.ndarray)):
        return _composition_stats(tuple(X))


@lru_cache()
def _composition_stats(X):
    """
    Calculate the CompositionStatistics for the input X. (see ..utils for more
    details).
    This is cached to avoid recalculation for the same input.
    Args:
        X: tuple

    Returns:
        (dict): composition_stats
    """
    stats = composition_statistics(X)
    return stats


class NumberOfCompositions(MetaFeature):
    """
    Number of compositions in the input dataset.
    """

    def calc(self, X, y=None):
        return len(X)


class PercentOfAllMetal(MetaFeature):
    """
    Percent of all_metal compositions in the input dataset.
    """

    def calc(self, X, y=None):
        stats = composition_stats(X)
        num = sum([1 if stat["major_composition_category"] == 1 else 0
                   for stat in stats.values()])
        return num / len(stats)


class PercentOfMetalNonmetal(MetaFeature):
    """
    Percent of metal_nonmetal compositions in the input dataset.
    """

    def calc(self, X, y=None):
        stats = composition_stats(X)
        num = sum([1 if stat["major_composition_category"] == 2 else 0
                   for stat in stats.values()])
        return num / len(stats)


class PercentOfAllNonmetal(MetaFeature):
    """
    Percent of all_nonmetal compositions in the input dataset.
    """

    def calc(self, X, y=None):
        stats = composition_stats(X)
        num = sum([1 if stat["major_composition_category"] == 3 else 0
                   for stat in stats.values()])
        return num / len(stats)


class PercentOfContainTransMetal(MetaFeature):
    """
    Percent of compositions containing transition_metal in the input dataset.
    """

    def calc(self, X, y=None):
        stats = composition_stats(X)
        num = sum([1 if 1 in stat["el_types_reduced"] else 0
                   for stat in stats.values()])
        return num / len(stats)


class NumberOfDifferentElements(MetaFeature):
    """
    Number of different elements present in the input dataset.
    """

    def calc(self, X, y=None):
        stats = composition_stats(X)
        elements = set()
        for stat in stats.values():
            elements = elements.union(set(stat["elements"]))
        return len(elements)


class AvgNumberOfElements(MetaFeature):
    """
    Average number of element of all compositions in the input dataset.
    """

    def calc(self, X, y=None):
        stats = composition_stats(X)
        nelements_sum = sum([stat["n_elements"] for stat in stats.values()])
        return nelements_sum / len(stats)


class MaxNumberOfElements(MetaFeature):
    """
    Maximum number of element number of all compositions in the input dataset.
    """

    def calc(self, X, y=None):
        stats = composition_stats(X)
        nelements_max = max([stat["n_elements"] for stat in stats.values()])
        return nelements_max


class MinNumberOfElements(MetaFeature):
    """
    Minimum number of element number of all compositions in the input dataset.
    """

    def calc(self, X, y=None):
        stats = composition_stats(X)
        nelements_min = min([stat["n_elements"] for stat in stats.values()])
        return nelements_min


# structure-related metafeatures
def structure_stats(X):
    """
    Transform the input X to immutable IStructure to use caching and call
    _structure_stats .
    Args:
        X: iterable structures, can be pymatgen Structure or IStructure objects

    Returns:
        _structure_stats
    """
    X_struct = list()
    for structure in X:
        if isinstance(structure, Structure):
            X_struct.append(IStructure.from_sites(structure))
        elif isinstance(structure, IStructure):
            X_struct.append(structure)
    return _structure_stats(tuple(X_struct))


@lru_cache()
def _structure_stats(X):
    """
    Calculate the StructureStatistics for the input X. (see ..utils for more
    details).
    This is cached to avoid recalculation for the same input.
    Args:
        X: tuple

    Returns:
        (dict): structure_stats
    """
    stats = structure_statistics(X)
    return stats


class NumberOfStructures(MetaFeature):
    """
    Number of structures in the input dataset.
    """

    def calc(self, X, y=None):
        return len(X)


class PercentOfOrderedStructures(MetaFeature):
    """
    Percent of ordered structures in the input dataset.
    """

    def calc(self, X, y=None):
        stats = structure_stats(X)
        num = sum([1 if stat["is_ordered"] else 0 for stat in stats.values()])
        return num / len(stats)


class AvgNumberOfSites(MetaFeature):
    """
    Average number of sites of all structures in the input dataset.
    """

    def calc(self, X, y=None):
        stats = structure_stats(X)
        nsites_sum = sum([stat["n_sites"] for stat in stats.values()])
        return nsites_sum / len(stats)


class MaxNumberOfSites(MetaFeature):
    """
    Maximum number of sites of all structures in the input dataset.
    """

    def calc(self, X, y=None):
        stats = structure_stats(X)
        nsites_max = max([stat["n_sites"] for stat in stats.values()])
        return nsites_max


class NumberOfDifferentElementsInStructure(MetaFeature):
    """
    Number of different elements present in all structures of the input dataset.
    """

    def calc(self, X, y=None):
        elements = set()
        for struct in X:
            c = Composition(struct.formula)
            els = [X.symbol for X in c.elements]
            elements = elements.union(set(els))
        return len(elements)


composition_mfs_dict = \
    {"number_of_compositions": NumberOfCompositions(),
     "percent_of_all_metal": PercentOfAllMetal(),
     "percent_of_metal_nonmetal": PercentOfMetalNonmetal(),
     "percent_of_all_nonmetal": PercentOfAllNonmetal(),
     "percent_of_contain_trans_metal": PercentOfContainTransMetal(),
     "number_of_different_elements": NumberOfDifferentElements(),
     "avg_number_of_elements": AvgNumberOfElements(),
     "max_number_of_elements": MaxNumberOfElements(),
     "min_number_of_elements": MinNumberOfElements()}

structure_mfs_dict = \
    {"number_of_structures": NumberOfStructures(),
     "percent_of_ordered_structures": PercentOfOrderedStructures(),
     "avg_number_of_sites": AvgNumberOfSites(),
     "max_number_of_sites": MaxNumberOfSites(),
     "number_of_different_elements_in_structures":
         NumberOfDifferentElementsInStructure()}


_supported_mfs_types = ("composition", "structure")


def dataset_metafeatures(df, **mfs_kwargs):
    """
    Given a dataset as a dataframe, calculate pre-defined metafeatures.
    (see ..metafeatures for more details).
    Return metafeatures of the dataset organized in a dict:
        {"composition_metafeatures": {"number_of_compositions": 2024,
                                  "percent_of_all_metal": 0.81,
                                  ...},
         "structure_metafeatures": {"number_of_structures": 2024,
                                  "percent_of_ordered_structures": 0.36,
                                  ...}}
    if there is no corresponding column in the dataset, the value is None.

    These dataset metafeatures will be used in FeaturizerMetaSelector to remove
    some featurizers that definitely do not work for this dataset (returning
    nans more than the allowed max_na_frac).
    Args:
        df: input dataset as pd.DataFrame
        mfs_kwargs: kwargs for _composition/structure_metafeatures

    Returns:
        (dict): {"composition_metafeatures": composition_mfs/None,
                 "structure_metafeatures": structure_mfs/None}
        """
    dataset_mfs = dict()
    for mfs_type in _supported_mfs_types:
        input_col = mfs_kwargs.get("{}_col".format(mfs_type), mfs_type)
        mfs_func = getattr(sys.modules[__name__],
                           "_{}_metafeatures".format(mfs_type), None)
        dataset_mfs.update(mfs_func(df, input_col)
                           if mfs_func is not None else {})

    return dataset_mfs


def _composition_metafeatures(df, composition_col="composition"):
    """
    Calculate composition-based metafeatures of the dataset.
    Args:
        df: input dataset as pd.DataFrame
        composition_col(str): column name for compositions

    Returns:
        (dict): {"composition_metafeatures": mfs/None}
    """
    if composition_col in df.columns:
        mfs = dict()
        for mf, mf_class in composition_mfs_dict.items():
            mfs[mf] = mf_class.calc(df[composition_col])
        return {"composition_metafeatures": mfs}
    else:
        return {"composition_metafeatures": None}


def _structure_metafeatures(df, structure_col="structure"):
    """
    Calculate structure-based metafeatures of the dataset.
    Args:
        df: input dataset as pd.DataFrame
        structure_col(str): column name in the df for structures, as pymatgen
            IStructure or Structure

    Returns:
        (dict): {"structure_metafeatures": mfs/None}
    """
    if structure_col in df.columns:
        mfs = dict()
        for mf, mf_class in structure_mfs_dict.items():
            mfs[mf] = mf_class.calc(df[structure_col])
        return {"structure_metafeatures": mfs}
    else:
        return {"structure_metafeatures": None}


class FeaturizerMetaSelector:
    """
    Given a dataset as a dataframe, heuristically customize the featurizers.
    Currently only support removing definitely useless featurizers.
    Cannot recommend featurizers based on the target now.
    """

    def __init__(self, max_na_frac=0.05):
        self.max_na_frac = max_na_frac
        self.dataset_mfs = None
        self.excludes = None

    @staticmethod
    def composition_featurizer_excludes(mfs, max_na_frac=0.05):
        """
        Determine the composition featurizers that are definitely do not work
        for this dataset (returning nans more than the allowed max_na_frac).
        Args:
            mfs: composition_metafeatures
            max_na_frac: max percent of nans allowed for the feature columns

        Returns:
            ([str]): list of removable composition featurizers
        """
        excludes = list()
        try:
            if mfs["percent_of_all_nonmetal"] > max_na_frac:
                excludes.extend(["Miedema",
                                 "YangSolidSolution"])

            if mfs["percent_of_contain_trans_metal"] < (1 - max_na_frac):
                excludes.extend(["TMetalFraction",
                                 "Miedema",
                                 "YangSolidSolution"])

            if mfs["percent_of_all_metal"] > max_na_frac:
                excludes.extend(["CationProperty",
                                 "OxidationStates",
                                 "ElectronAffinity",
                                 "ElectronegativityDiff",
                                 "IonProperty"])
        except KeyError:
            warnings.warn("The metafeature dict does not contain all the "
                          "metafeatures for filtering featurizers for the "
                          "compositions! Please call DatasetMetaFeatures first"
                          "to derive the metafeature dict.")

        return list(set(excludes))

    @staticmethod
    def structure_featurizer_excludes(mfs, max_na_frac=0.05):
        """
        Determine the structure featurizers that are definitely do not work
        for this dataset (returning nans more than the allowed max_na_frac).
        Args:
            mfs: structure_metafeatures
            max_na_frac: max percent of nans allowed for the feature columns

        Returns:
            ([str]): list of removable structure featurizers
        """
        excludes = list()
        try:
            if mfs["percent_of_ordered_structures"] < (1 - max_na_frac):
                excludes.extend(["GlobalSymmetryFeatures"])

        except KeyError:
            warnings.warn("The metafeature dict does not contain all the"
                          "metafeatures for filtering featurizers for the "
                          "structures! Please call DatasetMetaFeatures first"
                          "to derive the metafeature dict.")
        return list(set(excludes))

    def auto_excludes(self, df, **mfs_kwargs):
        """
        Automatically determine a list of removable featurizers based on
        metafeatures for all _supported_mfs_types.
        Args:
            auto_mfs_kwargs: kwargs for auto_metafeatures in DatasetMetafeatures

        Returns:
            ([str]): list of removable featurizers
        """
        auto_excludes = list()
        self.dataset_mfs = dataset_metafeatures(df, **mfs_kwargs)
        for mfs_type in _supported_mfs_types:
            mfs = self.dataset_mfs.get("{}_metafeatures".format(mfs_type))
            if mfs is not None:
                exclude_fts = getattr(self,
                                      "{}_featurizer_excludes".format(mfs_type),
                                      None)
                auto_excludes.extend(exclude_fts(mfs, self.max_na_frac)
                                     if exclude_fts is not None else [])

        self.excludes = list(set(auto_excludes))
        return self.excludes
