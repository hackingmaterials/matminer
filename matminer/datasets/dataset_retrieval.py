import os

from matminer.datasets.utils import _load_dataset_dict, _get_data_home, \
    _validate_dataset
from matminer.utils.io import load_dataframe_from_json

__author__ = "Kyle Bystrom <kylebystrom@berkeley.edu>, " \
             "Anubhav Jain <ajain@lbl.gov>" \
             "Daniel Dopp <dbdopp@lbl.gov>"

_dataset_dict = None


def load_dataset(name, data_home=None, download_if_missing=True):
    """
    Loads a dataframe containing the dataset specified with the 'name' field.

    Dataset file is stored/loaded from data_home if specified, otherwise at
    the MATMINER_DATA environment variable if set or at matminer/datasets
    by default.

    Args:
        name (str): keyword specifying what dataset to load, run
            matminer.datasets.get_available_datasets() for options

        data_home (str): path to folder to look for dataset file

        download_if_missing (bool): whether to download the dataset if is not
            found on disk

        include_metadata (bool): optional argument for some datasets with
            metadata fields

        system (list, str): argument for glass_ternary_hipt dataset,
            determines which subset of the dataset to load: "CoFeZr",
            "CoTiZr", "CoVZr","FeTiNb" or a list of these systems e.g.
            ["CoFeZr", "CoVZr"] or "all"

        return_lumo (bool): argument for double_perovskites_gap dataset,
            if True will return a second dataframe of the lowest unoccupied
            molecular orbital (LUMO) energy levels in eV.

        room_temperature (bool): argument for citrine_thermal_conductivity
            dataset, if selected only returns dataset items processed at room
            temperature.

        processing (str): argument for glass_ternary_landolt dataset, what type
            of processing to filter the dataset by, valid arguments are
            sputtering and meltspin. Default returns both methods

        unique_composition (bool): argument for glass_ternary_landolt dataset,
            Whether or not to combine items with different sources but the same
            composition, True by default

    Returns: (pd.DataFrame,
              tuple -> (pd.DataFrame, pd.DataFrame) if return_lumo = True)
    """
    global _dataset_dict

    if _dataset_dict is None:
        _dataset_dict = _load_dataset_dict()

    if name not in _dataset_dict:
        error_string = "Unrecognized dataset name: {}. \n" \
                       "Use matminer.datasets.get_available_datasets() " \
                       "to see a list of currently available " \
                       "datasets".format(name)

        # Very simple attempt to match unrecognized keyword to existing
        # dataset names in an attempt to give the user immediate feedback
        possible_matches = [
            x for x in _dataset_dict.keys() if name.lower() in x.lower()
        ]

        if possible_matches:
            error_string += "\nCould you have been looking for these similar " \
                            "matches?:\n{}".format(possible_matches)

        raise ValueError(error_string)

    dataset_metadata = _dataset_dict[name]
    data_path = os.path.join(_get_data_home(data_home),
                             name + "." + dataset_metadata['file_type'])
    _validate_dataset(data_path, dataset_metadata['url'],
                      dataset_metadata['hash'], download_if_missing)

    df = load_dataframe_from_json(data_path)

    return df


def load_elastic_tensor(version="2015", include_metadata=False, data_home=None,
                        download_if_missing=True):
    df = load_dataset("elastic_tensor" + "_" + version, data_home,
                      download_if_missing)

    if not include_metadata:
        df = df.drop(['cif', 'kpoint_density', 'poscar'], axis=1)

    return df


def load_piezoelectric_tensor(include_metadata=False, data_home=None,
                              download_if_missing=True):
    df = load_dataset("piezoelectric_tensor", data_home, download_if_missing)

    if not include_metadata:
        df = df.drop(['cif', 'meta', 'poscar'], axis=1)

    return df


def load_dielectric_constant(include_metadata=False, data_home=None,
                             download_if_missing=True):
    df = load_dataset("dielectric_constant", data_home, download_if_missing)

    if not include_metadata:
        df = df.drop(['cif', 'meta', 'poscar'], axis=1)

    return df


def load_glass_ternary_landolt(processing="all", unique_composition=True,
                               data_home=None, download_if_missing=True):
    df = load_dataset("glass_ternary_landolt", data_home, download_if_missing)

    if processing in {"meltspin", "sputtering"}:
        df = df[df["processing"] == processing]

    if unique_composition:
        df = df.groupby("formula").max().reset_index()

    return df


def load_double_perovskites_gap(return_lumo=False, data_home=None,
                                download_if_missing=True):
    df = load_dataset("double_perovskites_gap")

    if return_lumo:
        lumo = load_dataset("double_perovskites_gap_lumo", data_home,
                            download_if_missing)
        return df, lumo

    return df


def load_glass_ternary_hipt(system="all", data_home=None,
                            download_if_missing=True):
    df = load_dataset("glass_ternary_hipt", data_home, download_if_missing)

    if system != "all":
        if isinstance(system, str):
            system = [system]

        for item in system:
            if item not in {"CoFeZr", "CoTiZr", "CoVZr", "FeTiNb"}:
                raise AttributeError("some of the system list {} are not "
                                     "in this dataset". format(system))
        df = df[df["system"].isin(system)]

    return df


def load_citrine_thermal_conductivity(room_temperature=True, data_home=None,
                                      download_if_missing=True):
    df = load_dataset("citrine_thermal_conductivity", data_home,
                      download_if_missing)

    if room_temperature:
        df = df[df['k_condition'].isin(['room temperature',
                                        'Room temperature',
                                        'Standard',
                                        '298', '300'])]
    return df.drop(['k-units', 'k_condition', 'k_condition_units'], axis=1)


def get_available_datasets(print_datasets=True, print_descriptions=True,
                           sort_method='alphabetical'):
    """
    Function for retrieving the datasets available within matminer.

    Args:
        print_datasets (bool): Whether to, along with returning a
            list of dataset names, also print info on each dataset

        print_descriptions (bool): Whether to print the description of the
            dataset along with the name. Ignored if print_datasets is False

        sort_method (str): By what metric to sort the datasets when retrieving
            their information.

            alphabetical: sorts by dataset name,
            num_entries: sorts by number of dataset entries

    Returns: (list)
    """
    global _dataset_dict

    if _dataset_dict is None:
        _dataset_dict = _load_dataset_dict()

    if sort_method not in {"alphabetical", "num_entries"}:
        raise ValueError("Error, unsupported sorting metric {}"
                         " see docs for options".format(sort_method))

    if sort_method == 'num_entries':
        dataset_names = sorted(_dataset_dict.keys(),
                               key=lambda x: _dataset_dict[x]["num_entries"],
                               reverse=True)
    else:
        dataset_names = sorted(_dataset_dict.keys())

    # If checks done before for loop to avoid unnecessary repetitive evaluation
    if print_datasets and print_descriptions:
        for name in dataset_names:
            # Printing blank line with sep=\n to give extra line break
            print(name, _dataset_dict[name]["description"], "", sep="\n")
    elif print_datasets:
        for name in dataset_names:
            print(name)

    return dataset_names


def get_dataset_attribute(dataset_name, attrib_key):
    """
    Helper function for getting generic attributes of the dataset

    Args:
        dataset_name (str): Name of the dataset querying info from

        attrib_key (str): Name of attribute to pull

    Returns: Dataset attribute
    """
    # Load the dictionary into a global variable, keep around for future access
    global _dataset_dict

    if _dataset_dict is None:
        _dataset_dict = _load_dataset_dict()

    return _dataset_dict[dataset_name][attrib_key]


def get_dataset_citations(dataset_name):
    """
    Convenience function for getting dataset citations

    Args:
        dataset_name (str): name of the dataset being queried

    Returns: (list)
    """
    return get_dataset_attribute(dataset_name, 'bibtex_refs')


def get_dataset_reference(dataset_name):
    """
    Convenience function for getting dataset reference

    Args:
        dataset_name (str): name of the dataset being queried

    Returns: (str)
    """
    return get_dataset_attribute(dataset_name, 'reference')


def get_dataset_description(dataset_name):
    """
    Convenience function for getting dataset description

    Args:
        dataset_name (str): name of the dataset being queried

    Returns: (str)
    """
    return get_dataset_attribute(dataset_name, 'description')


def get_dataset_num_entries(dataset_name):
    """
    Convenience function for getting dataset number of entries

    Args:
        dataset_name (str): name of the dataset being queried

    Returns: (int)
    """
    return get_dataset_attribute(dataset_name, 'num_entries')


def get_dataset_columns(dataset_name):
    """
    Convenience function for getting dataset column list

    Args:
        dataset_name (str): name of the dataset being queried

    Returns: (list)
    """
    return list(get_dataset_attribute(dataset_name, 'columns').keys())


def get_dataset_column_description(dataset_name, dataset_column):
    """
    Convenience function for getting dataset column description

    Args:
        dataset_name (str): name of the dataset being queried
        dataset_column (str): name of the column to get description from

    Returns: (str)
    """
    return get_dataset_attribute(dataset_name, 'columns')[dataset_column]
