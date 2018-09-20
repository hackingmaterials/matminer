import json

from monty.json import MontyDecoder
from monty.dev import deprecated

import pandas as pd
from pymatgen import Composition
from pymatgen.core.structure import IStructure


@deprecated(message="matminer.utils.conversions.str_to_composition is "
                    "deprecated and will be removed in December 2018. Please use"
                    " the matminer.featurizers.conversions.StrToComposition "
                    "Featurizer instead")
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


@deprecated(message="matminer.utils.conversions.structure_to_composition is "
                    "deprecated and will be removed in December 2018. Please use"
                    " the matminer.featurizers.conversions.StructureToComposition"
                    " Featurizer instead")
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


@deprecated(message="matminer.utils.conversions.structure_to_istructure is "
                    "deprecated and will be removed in December 2018. Please use"
                    " the matminer.featurizers.conversions.StructureToIStructure"
                    " Featurizer instead")
def structure_to_istructure(series):
    """Convert a pymatgen Structure to an immutable IStructure object

    Useful if you are using features that employ caching

    Args:
        series (pd.Series): Series with pymatgen.Structure objects
    Returns:
        a pd.Series with the structures converted to IStructure
    """

    return series.map(IStructure.from_sites)


@deprecated(message="matminer.utils.conversions.dict_to_object is "
                    "deprecated and will be removed in December 2018. Please use"
                    " the matminer.featurizers.conversions.DictToObject"
                    " Featurizer instead")
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


@deprecated(message="matminer.utils.conversions.json_to_object is "
                    "deprecated and will be removed in December 2018. Please use"
                    " the matminer.featurizers.conversions.JsonToObject "
                    "Featurizer instead")
def json_to_object(series):
    """
        Decodes a json series to Python object series via MSON

        Args:
            series: a pd.Series with MSONable JSON components (string)

        Returns:
            a pd.Series with MSON objects, e.g. Structure objects
    """
    return series.map(lambda x: json.loads(x, cls=MontyDecoder))


@deprecated(message="matminer.utils.conversions.structure_to_oxidstructure is "
                    "deprecated and will be removed in December 2018. Please use"
                    " the matminer.featurizers.conversions.StructureToOxidstructure"
                    " Featurizer instead")
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


@deprecated(message="matminer.utils.conversions.composition_to_oxidcomposition is "
                    "deprecated and will be removed in December 2018. Please use"
                    " the matminer.featurizers.conversions.CompositionToOxidcomposition"
                    " Featurizer instead")
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

