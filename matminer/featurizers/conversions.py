"""
This module defines featurizers that can convert between different data formats


Note that these featurizers do not produce machine learning-ready features.
Instead, they should be used to pre-process data, either through a standalone
transformation or as part of a Pipeline.
"""

import json

from monty.json import MontyDecoder

from pymatgen.ext.matproj import MPRester
from pymatgen.core.structure import IStructure
from pymatgen.core.composition import Composition
from pymatgen.io.ase import AseAtomsAdaptor

from matminer.featurizers.base import BaseFeaturizer


class ConversionFeaturizer(BaseFeaturizer):
    """
    Abstract class to perform data conversions.

    Featurizers subclassing this class do not produce machine learning-ready
    features but instead are used to pre-process data. As Featurizers,
    the conversion process can take advantage of the parallelisation implemented
    in ScikitLearn.

    Note that `feature_labels` are set dynamically and may depend on the column
    id of the data being featurized. As such, `feature_labels` may differ
    before and after featurization.

    ConversionFeaturizers differ from other Featurizers in that the user can
    can specify the column in which to write the converted data. The output
    column is controlled through `target_col_id`. ConversionFeaturizers also
    have the ability to overwrite data in existing columns. This is
    controlled by the `overwrite_data` option. "in place" conversion of data can
    be achieved by setting `target_col_id=None` and `overwrite_data=True`. See
    the docstring below for more details.

    Args:
        target_col_id (str or None): The column in which the converted data will
            be written. If the column already exists then an error will be
            thrown unless `overwrite_data` is set to `True`. If `target_col_id`
            begins with an underscore the data will be written to the column:
            `"{}_{}".format(col_id, target_col_id[1:])`, where `col_id` is the
            column being featurized. If `target_col_id` is set to None then
            the data will be written "in place" to the `col_id` column (this
            will only work if `overwrite_data=True`).
        overwrite_data (bool): Overwrite any data in `target_col_id` if it
            exists.
    """

    def __init__(self, target_col_id, overwrite_data):
        self._target_col_id = target_col_id
        self._overwrite_data = overwrite_data

    def featurize_dataframe(self, df, col_id, **kwargs):
        """Perform the data conversion and set the target column dynamically.

        `target_col_id`, and accordingly `feature_labels`, may depend on the
        column id of the data being featurized. As such, `target_col_id` is
        first set dynamically before the `BaseFeaturizer.featurize_dataframe()`
        super method is called.

        Args:
            df (Pandas.DataFrame): Dataframe containing input data.
            col_id (str or list of str): column label containing objects to
                featurize. Can be multiple labels if the featurize function
                requires multiple inputs.
            **kwargs: Additional keyword arguments that will be passed through
                to `BaseFeaturizer.featurize_dataframe()`.

        Returns:
            (Pandas.Dataframe): The updated dataframe.
        """
        # now the col_id is known, we check if we need to update target_col_id
        # for multiindexes it is the last index that is updated
        target = self._target_col_id
        if isinstance(target, str) and target[0] == "_":
            if "multiindex" in kwargs and kwargs["multiindex"]:
                self._target_col_id = "{}_{}".format(col_id[-1], target[1:])
            else:
                self._target_col_id = "{}_{}".format(col_id, target[1:])
        elif target is None:
            self._target_col_id = col_id

        return super().featurize_dataframe(df, col_id, **kwargs)

    def featurize(self, *x):
        raise NotImplementedError("citations() is not defined!")

    def feature_labels(self):
        return [self._target_col_id]

    def citations(self):
        raise NotImplementedError("citations() is not defined!")

    def implementors(self):
        raise NotImplementedError("citations() is not defined!")


class StrToComposition(ConversionFeaturizer):
    """
    Utility featurizer to convert a string to a Composition

    The expected input is a composition in string form (e.g. "Fe2O3").

    Note that this Featurizer does not produce machine learning-ready features
    but instead can be applied to pre-process data or as part of a Pipeline.

    Args:
        reduce (bool): Whether to return a reduced
            `pymatgen.core.composition.Composition` object.
        target_col_id (str or None): The column in which the converted data will
            be written. If the column already exists then an error will be
            thrown unless `overwrite_data` is set to `True`. If `target_col_id`
            begins with an underscore the data will be written to the column:
            `"{}_{}".format(col_id, target_col_id[1:])`, where `col_id` is the
            column being featurized. If `target_col_id` is set to None then
            the data will be written "in place" to the `col_id` column (this
            will only work if `overwrite_data=True`).
        overwrite_data (bool): Overwrite any data in `target_column` if it
            exists.
    """

    def __init__(self, reduce=False, target_col_id="composition", overwrite_data=False):
        super().__init__(target_col_id, overwrite_data)
        self.reduce = reduce
        self._chunksize = 30

    def featurize(self, string_composition):
        """Convert a string to a pymatgen Composition.

        Args:
            string_composition (str): A chemical formula as a string (e.g.
                "Fe2O3").

        Returns:
            (`pymatgen.core.composition.Composition`): A composition object.
        """

        if self.reduce:
            return [Composition(string_composition).reduced_composition]
        else:
            return [Composition(string_composition)]

    def citations(self):
        return [
            "@article{ward_agrawal_choudary_wolverton_2016, title={A "
            "general-purpose machine learning framework for predicting "
            "properties of inorganic materials}, volume={2}, "
            "DOI={10.1038/npjcompumats.2017.28}, number={1}, journal={npj "
            "Computational Materials}, author={Ward, Logan and Agrawal, Ankit "
            "and Choudhary, Alok and Wolverton, Christopher}, year={2016}}"
        ]

    def implementors(self):
        return ["Anubhav Jain", "Alex Ganose"]


class StructureToComposition(ConversionFeaturizer):
    """
    Utility featurizer to convert a Structure to a Composition.

    The expected input is a `pymatgen.core.structure.Structure` object.

    Note that this Featurizer does not produce machine learning-ready features
    but instead can be applied to pre-process data or as part of a Pipeline.

    Args:
        reduce (bool): Whether to return a reduced Composition object.
        target_col_id (str or None): The column in which the converted data will
            be written. If the column already exists then an error will be
            thrown unless `overwrite_data` is set to `True`. If `target_col_id`
            begins with an underscore the data will be written to the column:
            `"{}_{}".format(col_id, target_col_id[1:])`, where `col_id` is the
            column being featurized. If `target_col_id` is set to None then
            the data will be written "in place" to the `col_id` column (this
            will only work if `overwrite_data=True`).
        overwrite_data (bool): Overwrite any data in `target_column` if it
            exists.
    """

    def __init__(self, reduce=False, target_col_id="composition", overwrite_data=False):
        super().__init__(target_col_id, overwrite_data)
        self.reduce = reduce
        self._overwrite = overwrite_data
        self._chunksize = 30

    def featurize(self, structure):
        """Convert a string to a pymatgen Composition.

        Args:
            structure (`pymatgen.core.structure.Structure`): A structure.

        Returns:
            (`pymatgen.core.composition.Composition`): A Composition object.
        """

        if self.reduce:
            return [structure.composition.reduced_composition]
        else:
            return [structure.composition]

    def citations(self):
        return [
            "@article{ward_agrawal_choudary_wolverton_2016, title={A "
            "general-purpose machine learning framework for predicting "
            "properties of inorganic materials}, volume={2}, "
            "DOI={10.1038/npjcompumats.2017.28}, number={1}, journal={npj "
            "Computational Materials}, author={Ward, Logan and Agrawal, Ankit "
            "and Choudhary, Alok and Wolverton, Christopher}, year={2016}}"
        ]

    def implementors(self):
        return ["Anubhav Jain", "Alex Ganose"]


class StructureToIStructure(ConversionFeaturizer):
    """
    Utility featurizer to convert a Structure to an immutable IStructure.

    This is useful if you are using features that employ caching.

    The expected input is a `pymatgen.core.structure.Structure` object.

    Note that this Featurizer does not produce machine learning-ready features
    but instead can be applied to pre-process data or as part of a Pipeline.

    Args:
        target_col_id (str or None): The column in which the converted data will
            be written. If the column already exists then an error will be
            thrown unless `overwrite_data` is set to `True`. If `target_col_id`
            begins with an underscore the data will be written to the column:
            `"{}_{}".format(col_id, target_col_id[1:])`, where `col_id` is the
            column being featurized. If `target_col_id` is set to None then
            the data will be written "in place" to the `col_id` column (this
            will only work if `overwrite_data=True`).
        overwrite_data (bool): Overwrite any data in `target_column` if it
            exists.
    """

    def __init__(self, target_col_id="istructure", overwrite_data=False):
        super().__init__(target_col_id, overwrite_data)
        self._overwrite = overwrite_data
        self._chunksize = 30

    def featurize(self, structure):
        """Convert a pymatgen Structure to an immutable IStructure,

        Args:
            structure (`pymatgen.core.structure.Structure`): A structure.

        Returns:
            (`pymatgen.core.structure.IStructure`): An immutable IStructure
                object.
        """
        return [IStructure.from_sites(structure)]

    def citations(self):
        return [
            "@article{ward_agrawal_choudary_wolverton_2016, title={A "
            "general-purpose machine learning framework for predicting "
            "properties of inorganic materials}, volume={2}, "
            "DOI={10.1038/npjcompumats.2017.28}, number={1}, journal={npj "
            "Computational Materials}, author={Ward, Logan and Agrawal, Ankit "
            "and Choudhary, Alok and Wolverton, Christopher}, year={2016}}"
        ]

    def implementors(self):
        return ["Logan Ward", "Alex Ganose"]


class DictToObject(ConversionFeaturizer):
    """
    Utility featurizer to decode a dict to Python object via MSON.

    Note that this Featurizer does not produce machine learning-ready features
    but instead can be applied to pre-process data or as part of a Pipeline.

    Args:
        target_col_id (str or None): The column in which the converted data will
            be written. If the column already exists then an error will be
            thrown unless `overwrite_data` is set to `True`. If `target_col_id`
            begins with an underscore the data will be written to the column:
            `"{}_{}".format(col_id, target_col_id[1:])`, where `col_id` is the
            column being featurized. If `target_col_id` is set to None then
            the data will be written "in place" to the `col_id` column (this
            will only work if `overwrite_data=True`).
        overwrite_data (bool): Overwrite any data in `target_column` if it
            exists.
    """

    def __init__(self, target_col_id="_object", overwrite_data=False):
        super().__init__(target_col_id, overwrite_data)
        self._chunksize = 30

    def featurize(self, dict_data):
        """Convert a string to a pymatgen Composition.

        Args:
            dict_data (dict): A MSONable dictionary. E.g. Produced from
                `pymatgen.core.structure.Structure.as_dict()`.

        Returns:
            (object): An object with the type specified by `dict_data`.
        """
        md = MontyDecoder()
        return [md.process_decoded(dict_data)]

    def citations(self):
        return [
            "@article{ward_agrawal_choudary_wolverton_2016, title={A "
            "general-purpose machine learning framework for predicting "
            "properties of inorganic materials}, volume={2}, "
            "DOI={10.1038/npjcompumats.2017.28}, number={1}, journal={npj "
            "Computational Materials}, author={Ward, Logan and Agrawal, Ankit "
            "and Choudhary, Alok and Wolverton, Christopher}, year={2016}}"
        ]

    def implementors(self):
        return ["Anubhav Jain", "Alex Ganose"]


class JsonToObject(ConversionFeaturizer):
    """
    Utility featurizer to decode json data to a Python object via MSON.

    Note that this Featurizer does not produce machine learning-ready features
    but instead can be applied to pre-process data or as part of a Pipeline.

    Args:
        target_col_id (str or None): The column in which the converted data will
            be written. If the column already exists then an error will be
            thrown unless `overwrite_data` is set to `True`. If `target_col_id`
            begins with an underscore the data will be written to the column:
            `"{}_{}".format(col_id, target_col_id[1:])`, where `col_id` is the
            column being featurized. If `target_col_id` is set to None then
            the data will be written "in place" to the `col_id` column (this
            will only work if `overwrite_data=True`).
        overwrite_data (bool): Overwrite any data in `target_column` if it
            exists.
    """

    def __init__(self, target_col_id="_object", overwrite_data=False):
        super().__init__(target_col_id, overwrite_data)
        self._chunksize = 30

    def featurize(self, json_data):
        """Convert a string to a pymatgen Composition.

        Args:
            json_data (dict): MSONable json data. E.g. Produced from
                `pymatgen.core.structure.Structure.to_json()`.

        Returns:
            (object): An object with the type specified by `json_data`.
        """
        return [json.loads(json_data, cls=MontyDecoder)]

    def citations(self):
        return [
            "@article{ward_agrawal_choudary_wolverton_2016, title={A "
            "general-purpose machine learning framework for predicting "
            "properties of inorganic materials}, volume={2}, "
            "DOI={10.1038/npjcompumats.2017.28}, number={1}, journal={npj "
            "Computational Materials}, author={Ward, Logan and Agrawal, Ankit "
            "and Choudhary, Alok and Wolverton, Christopher}, year={2016}}"
        ]

    def implementors(self):
        return ["Anubhav Jain", "Alex Ganose"]


class StructureToOxidStructure(ConversionFeaturizer):
    """
    Utility featurizer to add oxidation states to a pymatgen Structure.

    Oxidation states are determined using pymatgen's guessing routines.
    The expected input is a `pymatgen.core.structure.Structure` object.

    Note that this Featurizer does not produce machine learning-ready features
    but instead can be applied to pre-process data or as part of a Pipeline.

    Args:
        target_col_id (str or None): The column in which the converted data will
            be written. If the column already exists then an error will be
            thrown unless `overwrite_data` is set to `True`. If `target_col_id`
            begins with an underscore the data will be written to the column:
            `"{}_{}".format(col_id, target_col_id[1:])`, where `col_id` is the
            column being featurized. If `target_col_id` is set to None then
            the data will be written "in place" to the `col_id` column (this
            will only work if `overwrite_data=True`).
        overwrite_data (bool): Overwrite any data in `target_column` if it
            exists.
        return_original_on_error: If the oxidation states cannot be
            guessed and set to True, the structure without oxidation states will
            be returned. If set to False, an error will be thrown.
        **kwargs: Parameters to control the settings for
            `pymatgen.io.structure.Structure.add_oxidation_state_by_guess()`.
    """

    def __init__(self, target_col_id="structure_oxid", overwrite_data=False, return_original_on_error=False, **kwargs):
        super().__init__(target_col_id, overwrite_data)
        self.oxi_guess_params = kwargs
        self.return_original_on_error = return_original_on_error

    def featurize(self, structure):
        """Add oxidation states to a Structure using pymatgen's guessing routines.

        Args:
            structure (`pymatgen.core.structure.Structure`): A structure.

        Returns:
            (`pymatgen.core.structure.Structure`): A Structure object decorated
                with oxidation states.
        """
        els_have_oxi_states = [hasattr(s, "oxi_state") for s in structure.composition.elements]
        if all(els_have_oxi_states):
            return [structure]

        try:
            structure.add_oxidation_state_by_guess(**self.oxi_guess_params)
        except ValueError as e:
            if not self.return_original_on_error:
                raise e

        return [structure]

    def citations(self):
        return [
            "@article{ward_agrawal_choudary_wolverton_2016, title={A "
            "general-purpose machine learning framework for predicting "
            "properties of inorganic materials}, volume={2}, "
            "DOI={10.1038/npjcompumats.2017.28}, number={1}, journal={npj "
            "Computational Materials}, author={Ward, Logan and Agrawal, Ankit "
            "and Choudhary, Alok and Wolverton, Christopher}, year={2016}}"
        ]

    def implementors(self):
        return ["Anubhav Jain", "Alex Ganose"]


class CompositionToOxidComposition(ConversionFeaturizer):
    """
    Utility featurizer to add oxidation states to a pymatgen Composition.

    Oxidation states are determined using pymatgen's guessing routines.
    The expected input is a `pymatgen.core.composition.Composition` object.

    Note that this Featurizer does not produce machine learning-ready features
    but instead can be applied to pre-process data or as part of a Pipeline.

    Args:
        target_col_id (str or None): The column in which the converted data will
            be written. If the column already exists then an error will be
            thrown unless `overwrite_data` is set to `True`. If `target_col_id`
            begins with an underscore the data will be written to the column:
            `"{}_{}".format(col_id, target_col_id[1:])`, where `col_id` is the
            column being featurized. If `target_col_id` is set to None then
            the data will be written "in place" to the `col_id` column (this
            will only work if `overwrite_data=True`).
        overwrite_data (bool): Overwrite any data in `target_column` if it
            exists.
        coerce_mixed (bool): If a composition has both species containing
            oxid states and not containing oxid states, strips all of the
            oxid states and guesses the entire composition's oxid states.
        return_original_on_error: If the oxidation states cannot be
            guessed and set to True, the composition without oxidation states
            will be returned. If set to False, an error will be thrown.
        **kwargs: Parameters to control the settings for
            `pymatgen.io.structure.Structure.add_oxidation_state_by_guess()`.

    """

    def __init__(
        self,
        target_col_id="composition_oxid",
        overwrite_data=False,
        coerce_mixed=True,
        return_original_on_error=False,
        **kwargs,
    ):
        super().__init__(target_col_id, overwrite_data)
        self.oxi_guess_params = kwargs
        self.coerce_mixed = coerce_mixed
        self.return_original_on_error = return_original_on_error

    def featurize(self, comp):
        """Add oxidation states to a Structure using pymatgen's guessing routines.

        Args:
            comp (`pymatgen.core.composition.Composition`): A composition.

        Returns:
            (`pymatgen.core.composition.Composition`): A Composition object
                decorated with oxidation states.
        """
        els_have_oxi_states = [hasattr(s, "oxi_state") for s in comp.elements]

        if all(els_have_oxi_states):
            return [comp]

        elif any(els_have_oxi_states):
            if self.coerce_mixed:
                comp = comp.element_composition
            else:
                raise ValueError(
                    "Composition {} has a mix of species with "
                    "and without oxidation states. Please enable "
                    "coercion to all oxidation states with "
                    "coerce_mixed.".format(comp)
                )
        try:
            comp = comp.add_charges_from_oxi_state_guesses(**self.oxi_guess_params)
        except ValueError as e:
            if not self.return_original_on_error:
                raise e
        return [comp]

    def citations(self):
        return [
            "@article{ward_agrawal_choudary_wolverton_2016, title={A "
            "general-purpose machine learning framework for predicting "
            "properties of inorganic materials}, volume={2}, "
            "DOI={10.1038/npjcompumats.2017.28}, number={1}, journal={npj "
            "Computational Materials}, author={Ward, Logan and Agrawal, Ankit "
            "and Choudhary, Alok and Wolverton, Christopher}, year={2016}}"
        ]

    def implementors(self):
        return ["Anubhav Jain", "Alex Ganose", "Alex Dunn"]


class CompositionToStructureFromMP(ConversionFeaturizer):
    """
    Featurizer to get a Structure object from Materials Project using the
    composition alone. The most stable entry from Materials Project is selected,
    or NaN if no entry is found in the Materials Project.

    Args:
        target_col_id (str or None): The column in which the converted data will
            be written. If the column already exists then an error will be
            thrown unless `overwrite_data` is set to `True`. If `target_col_id`
            begins with an underscore the data will be written to the column:
            `"{}_{}".format(col_id, target_col_id[1:])`, where `col_id` is the
            column being featurized. If `target_col_id` is set to None then
            the data will be written "in place" to the `col_id` column (this
            will only work if `overwrite_data=True`).
        overwrite_data (bool): Overwrite any data in `target_column` if it
            exists.
        map_key (str): Materials API key

    """

    def __init__(self, target_col_id="structure", overwrite_data=False, mapi_key=None):
        super().__init__(target_col_id, overwrite_data)
        self.mpr = MPRester(mapi_key)
        self.set_n_jobs(1)

    def featurize(self, comp):
        """
        Get the most stable structure from Materials Project
        Args:
            comp (`pymatgen.core.composition.Composition`): A composition.

        Returns:
            (`pymatgen.core.structure.Structure`): A Structure object.
        """

        entries = self.mpr.get_data(comp.reduced_formula, prop="energy_per_atom")
        if len(entries) > 0:
            most_stable_entry = sorted(entries, key=lambda e: e["energy_per_atom"])[0]
            s = self.mpr.get_structure_by_material_id(most_stable_entry["material_id"])
            return [s]

        return [float("nan")]

    def citations(self):
        return [
            "@article{doi:10.1063/1.4812323, author = {Jain,Anubhav and Ong,"
            "Shyue Ping  and Hautier,Geoffroy and Chen,Wei and Richards, "
            "William Davidson  and Dacek,Stephen and Cholia,Shreyas "
            "and Gunter,Dan  and Skinner,David and Ceder,Gerbrand "
            "and Persson,Kristin A. }, title = {Commentary: The Materials "
            "Project: A materials genome approach to accelerating materials "
            "innovation}, journal = {APL Materials}, volume = {1}, number = "
            "{1}, pages = {011002}, year = {2013}, doi = {10.1063/1.4812323}, "
            "URL = {https://doi.org/10.1063/1.4812323}, "
            "eprint = {https://doi.org/10.1063/1.4812323}}",
            "@article{Ong2015, author = {Ong, Shyue Ping and Cholia, "
            "Shreyas and Jain, Anubhav and Brafman, Miriam and Gunter, Dan "
            "and Ceder, Gerbrand and Persson, Kristin a.}, doi = "
            "{10.1016/j.commatsci.2014.10.037}, issn = {09270256}, "
            "journal = {Computational Materials Science}, month = {feb}, "
            "pages = {209--215}, publisher = {Elsevier B.V.}, title = "
            "{{The Materials Application Programming Interface (API): A simple, "
            "flexible and efficient API for materials data based on "
            "REpresentational State Transfer (REST) principles}}, "
            "url = {http://linkinghub.elsevier.com/retrieve/pii/S0927025614007113}, "
            "volume = {97}, year = {2015} } ",
        ]

    def implementors(self):
        return ["Anubhav Jain"]


class PymatgenFunctionApplicator(ConversionFeaturizer):
    """
    Featurizer to run any function using on/from pymatgen primitives.

    For example, apply

        lambda structure: structure.composition.anonymized_formula

    To all rows in a dataframe.

    And return the results in the specified column.

    Args:
        func (function): Function object or lambda to pass the pmg primitive objects to.
        func_args (list): List of args to pass along with the pmg object to func.
        func_kwargs (dict): Dict of kwargs to pass along with the pmg object to func,
        target_col_id (str): Output column for the results. If not provided, the func name
            will be used.
        overwrite_data (bool): If True, will overwrite target_col_id even if there is
            data currently in that column
    """

    def __init__(self, func, func_args=None, func_kwargs=None, target_col_id=None, overwrite_data=False):

        if not callable(func):
            raise TypeError(f"Function {func} is not callable!")

        if not target_col_id:
            target_col_id = func.__name__

        super().__init__(target_col_id, overwrite_data)

        self.func = func
        self.func_args = func_args if func_args else []
        self.func_kwargs = func_kwargs if func_kwargs else {}

        # n_jobs must be set to 1 to avoid pickling errors
        self.set_n_jobs(1)

    def featurize(self, obj):
        return (self.func(obj, *self.func_args, **self.func_kwargs),)

    def implementors(self):
        return ["Alex Dunn"]


class ASEAtomstoStructure(ConversionFeaturizer):
    """
    Convert dataframes of ase structures to pymatgen structures for further use with
    matminer.

    Args:
        target_col_id (str): Column to place PMG structures.
        overwrite_data (bool): If True, will overwrite target_col_id even if there is
            data currently in that column
    """

    def __init__(self, target_col_id="PMG Structure from ASE Atoms", overwrite_data=False):
        super().__init__(target_col_id, overwrite_data)
        self.aaa = AseAtomsAdaptor()

    def featurize(self, ase_atoms):
        return (self.aaa.get_structure(ase_atoms),)

    def implementors(self):
        return ["Alex Dunn"]
