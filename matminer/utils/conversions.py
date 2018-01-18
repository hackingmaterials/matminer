import json

from monty.json import MontyDecoder
from pandas import Series

from pymatgen import Composition


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
        copy = Series(data=[x.copy() for x in series.tolist()],
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
