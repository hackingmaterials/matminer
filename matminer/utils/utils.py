import warnings

import pandas as pd


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