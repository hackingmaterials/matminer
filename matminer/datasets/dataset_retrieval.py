import os

from matminer.datasets.utils import _load_dataset_dict, _get_data_home, \
    _validate_dataset
from matminer.utils.io import load_dataframe_from_json

__author__ = "Kyle Bystrom <kylebystrom@berkeley.edu>, " \
             "Anubhav Jain <ajain@lbl.gov>" \
             "Daniel Dopp <dbdopp@lbl.gov>"


def load_dataset(name, data_home=None, download_if_missing=True,
                 include_metadata=False):
    """
    Loads a dataframe containing the dataset specified with the 'name' field.

    Dataset file is stored/loaded from data_home if specified, otherwise at
    the MATMINER_DATA environment variable if set or at matminer/datasets
    by default.

    Args:
        name (str): keyword specifying what dataset to load, run
            matminer.datasets.available_datasets() for options

        data_home (str): path to folder to look for dataset file

        download_if_missing (bool): whether to download the dataset if is not
            found on disk

        include_metadata (bool): optional argument for some datasets with
            metadata fields

    Returns: (pd.DataFrame)
    """
    dataset_dict = _load_dataset_dict()

    if name not in dataset_dict:
        error_string = "Unrecognized dataset name: {}. \n" \
                       "Use matminer.datasets.available_datasets() " \
                       "to see a list of currently available " \
                       "datasets".format(name)

        # Very simple attempt to match unrecognized keyword to existing
        # dataset names in an attempt to give the user immediate feedback
        possible_matches = [
            x for x in dataset_dict.keys() if name.lower() in x.lower()
        ]

        if possible_matches:
            error_string += "\nCould you have been looking for these similar " \
                            "matches?:\n{}".format(possible_matches)

        raise ValueError(error_string)

    dataset_metadata = dataset_dict[name]
    data_path = os.path.join(_get_data_home(data_home),
                             name + "." + dataset_metadata['file_type'])
    _validate_dataset(data_path, dataset_metadata['url'],
                      dataset_metadata['hash'], download_if_missing)

    df = load_dataframe_from_json(data_path)

    if not include_metadata:
        if name == "elastic_tensor_2015":
            df = df.drop(['cif', 'kpoint_density', 'poscar'], axis=1)

        elif name in {"piezoelectric_tensor", "dielectric_constant"}:
            df = df.drop(['cif', 'meta', 'poscar'], axis=1)

    return df


def available_datasets(print_datasets=True, print_descriptions=True,
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
    dataset_dict = _load_dataset_dict()

    if sort_method not in {"alphabetical", "num_entries"}:
        raise ValueError("Error, unsupported sorting metric {}"
                         " see docs for options".format(sort_method))

    if sort_method == 'num_entries':
        dataset_names = sorted(dataset_dict.keys(),
                               key=lambda x: dataset_dict[x]["num_entries"],
                               reverse=True)
    else:
        dataset_names = sorted(dataset_dict.keys())

    # If checks done before for loop to avoid unnecessary repetitive evaluation
    if print_datasets and print_descriptions:
        for name in dataset_names:
            # Printing blank line with sep=\n to give extra line break
            print(name, dataset_dict[name]["description"], "", sep="\n")
    elif print_datasets:
        for name in dataset_names:
            print(name)

    return list(dataset_names)
