"""
This module defines featurizers that can convert between different data formats


Note that these featurizers do not produce machine learning-ready features.
Instead, they should be used to pre-process data, either through a standalone
transformation or as part of a Pipeline.
"""

import json

from monty.json import MontyDecoder

from pymatgen.core.structure import IStructure
from pymatgen.core.composition import Composition

from matminer.featurizers.base import BaseFeaturizer


class ConversionFeaturizer(BaseFeaturizer):
    """Abstract class to perform data conversions.

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
                to `BseFeaturizer.featurize_dataframe()`.

        Returns:
            (Pandas.Dataframe): The updated dataframe.
        """

        # TODO: Figure out if this is possible/desired
        if 'multiiindex' in kwargs and kwargs['multiindex']:
            raise ValueError("ConversionFeaturizer does not support "
                             "multiindexing")

        # now the col_id is known, we check if we need to update target_col_id
        # for multiindexes it is the last index that is updated
        target = self._target_col_id
        if isinstance(target, str) and target[0] == '_':
            if 'multiindex' in kwargs and kwargs['multiindex']:
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
    """Utility featurizer to convert a string to a Composition

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

    def __init__(self, reduce=False, target_col_id='composition',
                 overwrite_data=False):
        super().__init__(target_col_id, overwrite_data)
        self.reduce = reduce

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
        return [(
            "@article{ward_agrawal_choudary_wolverton_2016, title={A "
            "general-purpose machine learning framework for predicting "
            "properties of inorganic materials}, volume={2}, "
            "DOI={10.1038/npjcompumats.2017.28}, number={1}, journal={npj "
            "Computational Materials}, author={Ward, Logan and Agrawal, Ankit "
            "and Choudhary, Alok and Wolverton, Christopher}, year={2016}}")]

    def implementors(self):
        return ["Anubhav Jain", "Alex Ganose"]


class StructureToComposition(ConversionFeaturizer):
    """Utility featurizer to convert a Structure to a Composition.

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

    def __init__(self, reduce=False, target_col_id='composition',
                 overwrite_data=False):
        super().__init__(target_col_id, overwrite_data)
        self.reduce = reduce
        self._overwrite = overwrite_data

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
        return [(
            "@article{ward_agrawal_choudary_wolverton_2016, title={A "
            "general-purpose machine learning framework for predicting "
            "properties of inorganic materials}, volume={2}, "
            "DOI={10.1038/npjcompumats.2017.28}, number={1}, journal={npj "
            "Computational Materials}, author={Ward, Logan and Agrawal, Ankit "
            "and Choudhary, Alok and Wolverton, Christopher}, year={2016}}")]

    def implementors(self):
        return ["Anubhav Jain", "Alex Ganose"]


class StructureToIStructure(ConversionFeaturizer):
    """Utility featurizer to convert a Structure to an immutable IStructure.

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

    def __init__(self, target_col_id='istructure', overwrite_data=False):
        super().__init__(target_col_id, overwrite_data)
        self._overwrite = overwrite_data

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
        return [(
            "@article{ward_agrawal_choudary_wolverton_2016, title={A "
            "general-purpose machine learning framework for predicting "
            "properties of inorganic materials}, volume={2}, "
            "DOI={10.1038/npjcompumats.2017.28}, number={1}, journal={npj "
            "Computational Materials}, author={Ward, Logan and Agrawal, Ankit "
            "and Choudhary, Alok and Wolverton, Christopher}, year={2016}}")]

    def implementors(self):
        return ["Logan Ward", "Alex Ganose"]


class DictToObject(ConversionFeaturizer):
    """Utility featurizer to decode a dict to Python object via MSON.

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

    def __init__(self, target_col_id='_object', overwrite_data=False):
        super().__init__(target_col_id, overwrite_data)

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
        return [(
            "@article{ward_agrawal_choudary_wolverton_2016, title={A "
            "general-purpose machine learning framework for predicting "
            "properties of inorganic materials}, volume={2}, "
            "DOI={10.1038/npjcompumats.2017.28}, number={1}, journal={npj "
            "Computational Materials}, author={Ward, Logan and Agrawal, Ankit "
            "and Choudhary, Alok and Wolverton, Christopher}, year={2016}}")]

    def implementors(self):
        return ["Anubhav Jain", "Alex Ganose"]


class JsonToObject(ConversionFeaturizer):
    """Utility featurizer to decode json data to a Python object via MSON.

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

    def __init__(self, target_col_id='_object', overwrite_data=False):
        super().__init__(target_col_id, overwrite_data)

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
        return [(
            "@article{ward_agrawal_choudary_wolverton_2016, title={A "
            "general-purpose machine learning framework for predicting "
            "properties of inorganic materials}, volume={2}, "
            "DOI={10.1038/npjcompumats.2017.28}, number={1}, journal={npj "
            "Computational Materials}, author={Ward, Logan and Agrawal, Ankit "
            "and Choudhary, Alok and Wolverton, Christopher}, year={2016}}")]

    def implementors(self):
        return ["Anubhav Jain", "Alex Ganose"]


class StructureToOxidStructure(ConversionFeaturizer):
    """Utility featurizer to add oxidation states to a pymatgen Structure.

    Oxidation states are determined using pymatgen's guessing routines.
    The expected input is a `pymatgen.core.structure.Structure` object.

    Note that this Featurizer does not produce machine learning-ready features
    but instead can be applied to pre-process data or as part of a Pipeline.

    Args:
        **kwargs: Parameters to control the settings for
            `pymatgen.io.structure.Structure.add_oxidation_state_by_guess()`.
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

    def __init__(self, target_col_id='structure_oxid', overwrite_data=False,
                 **kwargs):
        super().__init__(target_col_id, overwrite_data)
        self.oxi_guess_params = kwargs

    def featurize(self, structure):
        """Add oxidation states to a Structure using pymatgen's guessing routines.

        Args:
            structure (`pymatgen.core.structure.Structure`): A structure.

        Returns:
            (`pymatgen.core.structure.Structure`): A Structure object decorated
                with oxidation states.
        """
        structure.add_oxidation_state_by_guess(**self.oxi_guess_params)
        return [structure]

    def citations(self):
        return [(
            "@article{ward_agrawal_choudary_wolverton_2016, title={A "
            "general-purpose machine learning framework for predicting "
            "properties of inorganic materials}, volume={2}, "
            "DOI={10.1038/npjcompumats.2017.28}, number={1}, journal={npj "
            "Computational Materials}, author={Ward, Logan and Agrawal, Ankit "
            "and Choudhary, Alok and Wolverton, Christopher}, year={2016}}")]

    def implementors(self):
        return ["Anubhav Jain", "Alex Ganose"]


class CompositionToOxidComposition(ConversionFeaturizer):
    """Utility featurizer to add oxidation states to a pymatgen Composition.

    Oxidation states are determined using pymatgen's guessing routines.
    The expected input is a `pymatgen.core.composition.Composition` object.

    Note that this Featurizer does not produce machine learning-ready features
    but instead can be applied to pre-process data or as part of a Pipeline.

    Args:
        **kwargs: Parameters to control the settings for
            `pymatgen.io.structure.Structure.add_oxidation_state_by_guess()`.
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

    def __init__(self, target_col_id='composition_oxid', overwrite_data=False,
                 **kwargs):
        super().__init__(target_col_id, overwrite_data)
        self.oxi_guess_params = kwargs

    def featurize(self, comp):
        """Add oxidation states to a Structure using pymatgen's guessing routines.

        Args:
            comp (`pymatgen.core.composition.Composition`): A composition.

        Returns:
            (`pymatgen.core.composition.Composition`): A Composition object
                decorated with oxidation states.
        """
        return [comp.add_charges_from_oxi_state_guesses(
            **self.oxi_guess_params)]

    def citations(self):
        return [(
            "@article{ward_agrawal_choudary_wolverton_2016, title={A "
            "general-purpose machine learning framework for predicting "
            "properties of inorganic materials}, volume={2}, "
            "DOI={10.1038/npjcompumats.2017.28}, number={1}, journal={npj "
            "Computational Materials}, author={Ward, Logan and Agrawal, Ankit "
            "and Choudhary, Alok and Wolverton, Christopher}, year={2016}}")]

    def implementors(self):
        return ["Anubhav Jain", "Alex Ganose"]
