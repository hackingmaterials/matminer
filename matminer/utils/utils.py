""" Utility module """
import warnings

import numpy as np
import pandas as pd
from numpy.linalg import pinv
from pymatgen.core import Composition, Element


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
        raise IndexError(
            "An input dataframe of 2+ levels cannot be used for"
            "multiindexed Matminer featurization without coercion "
            "to 2 levels."
        )


def get_elem_in_data(df, as_pure=False):
    """
    Looks for all elements present in the compounds forming the index of a dataframe.

    Args:
        df: DataFrame containing the compounds formula (as strings) as index
        as_pure: if True, consider only the pure compounds

    Returns:
        List of elements (str)
    """
    elems_in_df = []
    elems_not_in_df = []

    # Find the elements in the data, as pure or not
    if as_pure:
        for elem in Element:
            if elem.name in df.index:
                elems_in_df.append(elem.name)
    else:
        for elem in Element:
            for compound in df.index.to_list():
                if elem in Composition(compound) and elem.name not in elems_in_df:
                    elems_in_df.append(elem.name)

    # Find the elements not in the data
    for elem in Element:
        if elem.name not in elems_in_df:
            elems_not_in_df.append(elem.name)

    return elems_in_df, elems_not_in_df


def get_pseudo_inverse(df_init, cols=None):
    """
    Computes the pseudo-inverse matrix of a dataframe containing properties for multiple compositions.
    From a compositions matrix (containing the elemental fractions of each element present in the data),
    and a property column (or multiple properties gathered in a matrix), the pseudo-inverse column
    (or matrix if multiple properties are present) is defined by

    compositions * pseudo-inverse = property

    The pseudo-inverse coefficients are therefore average contributions of each element present in the data to the
    property of the compounds containing this element (in a least-square fit manner).
    This allows to take many compounds-property into consideration.

    Note that the pseudo-inverse coefficients do not represent the property of single elements in their pure form:
    negative values may appear for single elements and for physically-positive properties, but this only reflects
    the fact that the presence of the element in compounds generally decreases the value of the property.

    For elements that are not present in compounds of the data, nan are returned.

    Args:
        df_init: DataFrame with a column named "Composition" containing compositions
                 (anything that can be turned into a Pymatgen Composition object),
                 and other columns containing properties to be inversed.
        cols: list of columns of the dataframe giving the features to be pseudo-inversed.

    Returns:
        DataFrame with the pseudo-inverse coefficients for all elements present in the initial compositions and all
        properties.
    """
    df = df_init.copy()

    if "Composition" not in df.columns:
        raise LookupError("The DataFrame should contain a column named Composition")

    if cols is None:
        cols = list(df.columns)
        cols.remove("Composition")

    data = df[cols]
    data.index = df["Composition"].map(lambda s: str(s))

    elems_in_df, elems_not_in_df = get_elem_in_data(data, as_pure=False)
    n_elem_tot = len(elems_in_df)

    # Initialize the matrix
    A = np.zeros([len(data), n_elem_tot])

    compos = df["Composition"]
    for i, comp in enumerate(compos):
        comp = Composition(comp)
        for j in range(n_elem_tot):
            if elems_in_df[j] in comp:
                A[i, j] = comp.get_atomic_fraction(elems_in_df[j])

    pi_A = pinv(A)

    res_pi = pi_A @ data.values
    res_pi = np.vstack([res_pi, np.nan * np.ones([len(elems_not_in_df), len(df.T) - 1])])

    df_pi = pd.DataFrame(res_pi, columns=cols, index=pd.Index(elems_in_df + elems_not_in_df))
    # Handle the case of hydrogen, deuterium, and tritium
    # If all are present, there contributions are summed
    # and given to hydrogen only. Others are removed.
    if all(e in df_pi.index for e in ["H", "D", "T"]):
        df_pi.loc["H"] = df_pi.loc["H"] + df_pi.loc["D"] + df_pi.loc["T"]
        df_pi.drop(index=["T", "D"], inplace=True)

    return df_pi
