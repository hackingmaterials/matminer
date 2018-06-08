import json
import warnings
from monty.json import MontyDecoder
import pandas as pd
from pymatgen import Composition
from pymatgen.core.structure import IStructure


def homogenize_multiindex(df, default_key, coerce=False):
    """
    Homogenizes a dataframe column index to a 2-level multiindex.

    Args:
        df (pandas DataFrame): A dataframe
        default_key (str): The key to use when a single Index must be converted
            to a 2-level index. This key is then used as a parent of all
            keys present in the original 1-level index.
        coerce (bool): If True, try to force a 2+ level multiindex to a 2-level
            multiindex.

    Returns:
        df (pandas DataFrame): A dataframe with a 2-layer multiindex.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        cols = pd.MultiIndex.from_product(([default_key], df.columns.values))
        df.columns = cols
        return df
    elif df.columns.nlevels == 2:
        return df
    elif coerce:
        # Drop levels lower than the base column indices
        warnings.warn("Multiindex has nlevels more than 2! Coercing...")
        l1 = df.columns.get_level_values(df.columns.nlevels - 1)
        l2 = df.columns.get_level_values(df.columns.nlevels - 2)
        cols = pd.MultiIndex.from_arrays((l2, l1))
        df.columns = cols
        return df
    else:
        raise IndexError("An input dataframe of 2+ levels cannot be used for"
                         "multiindexed Matminer featurization without coercion "
                         "to 2 levels.")

def str_to_composition(series, reduce=False):
    """
    Converts a String series to a Composition series

    Args:
        series: a pd.Series with str components, e.g. "Fe2O3"
        reduce: (bool) whether to return a reduced Composition

    Returns:
        a pd.Series with pymatgen Composition components
    """
    if reduce:
        return series.map(lambda x: Composition(x).reduced_composition)

    return series.map(Composition)


def structure_to_composition(series, reduce=False):
    """
    Converts a Structure series to a Composition series

    Args:
        series: a pd.Series with pymatgen.Structure components
        reduce: (bool) whether to return a reduced Composition

    Returns:
        a pd.Series with pymatgen Composition components
    """
    if reduce:
        return series.map(lambda x: x.composition.reduced_composition)

    return series.map(lambda x: x.composition)


def structure_to_istructure(series):
    """Convert a pymatgen Structure to an immutable IStructure object

    Useful if you are using features that employ caching

    Args:
        series (pd.Series): Series with pymatgen.Structure objects
    Returns:
        a pd.Series with the structures converted to IStructure
    """

    return series.map(IStructure.from_sites)


def dict_to_object(series):
    """
        Decodes a dict Series to Python object series via MSON

        Args:
            series: a pd.Series with MSONable dict components, e.g.,
            pymatgen Structure.as_dict()

        Returns:
            a pd.Series with MSON objects, e.g. Structure objects
    """
    md = MontyDecoder()
    return series.map(md.process_decoded)


def json_to_object(series):
    """
        Decodes a json series to Python object series via MSON

        Args:
            series: a pd.Series with MSONable JSON components (string)

        Returns:
            a pd.Series with MSON objects, e.g. Structure objects
    """
    return series.map(lambda x: json.loads(x, cls=MontyDecoder))


def structure_to_oxidstructure(series, inplace=False, **kwargs):
    """
    Adds oxidation states to a Structure using pymatgen's guessing routines

    Args:
        series: a pd.Series with Structure object components
        inplace: (bool) whether to override original Series (this is faster)
        **kwargs: parameters to control to Structure.add_oxidation_state_by_guess

    Returns:
        a pd.Series with oxidation state Structure object components
    """
    if inplace:
        series.map(lambda s: s.add_oxidation_state_by_guess(**kwargs))
    else:
        copy = pd.Series(data=[x.copy() for x in series.tolist()],
                         index=series.index, dtype=series.dtype)
        copy.map(lambda s: s.add_oxidation_state_by_guess(**kwargs))
        return copy


def composition_to_oxidcomposition(series, **kwargs):
    """
    Adds oxidation states to a Composition using pymatgen's guessing routines

    Args:
        series: a pd.Series with Composition object components
        **kwargs: parameters to control Composition.oxi_state_guesses()

    Returns:
        a pd.Series with oxidation state Composition object components
    """

    return series.map(lambda c: c.add_charges_from_oxi_state_guesses(**kwargs))
