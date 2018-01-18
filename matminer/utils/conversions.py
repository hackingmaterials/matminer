import json

from monty.json import MontyDecoder
from pandas import Series

from pymatgen import Composition


def str_to_composition(series):
    """
    Converts a String series to a Composition series

    Args:
        series: a pd.Series with str components, e.g. "Fe2O3"

    Returns:
        a pd.Series with pymatgen Composition components
    """
    return series.map(Composition)


def structure_to_composition(series, reduce=False):
    """
    Converts a Structure series to a Composition series

    Args:
        series: a pd.Series with pymatgen.Structure components

    Returns:
        a pd.Series with pymatgen Composition components
    """
    if reduce:
        return series.map(lambda x: x.composition.reduced_composition)
    else:
        return series.map(lambda x: x.composition)


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


def struct_to_oxidstruct(series, inplace=False, **kwargs):
    """
    Adds oxidation states to a structure using pymatgen's guessing routines

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
        copy = Series(data=[x.copy() for x in series.tolist()],
                      index=series.index, dtype=series.dtype)
        copy.map(lambda s: s.add_oxidation_state_by_guess(**kwargs))
        return copy