import ast
import argparse
from os import listdir, makedirs
from os.path import isdir, join, expanduser, basename
from collections import defaultdict

import numpy as np
import pandas
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.core.structure import Structure

from matminer.datasets.utils import _get_file_sha256_hash
from matminer.utils.io import store_dataframe_as_json

__author__ = "Daniel Dopp <dbdopp@lbl.gov>"

"""Each _preprocess_* function acts as a preprocessor for a dataset with a 
given name. One should be written whenever a new dataset is being added to
matminer"""


def _preprocess_boltztrap_mp(df):
    df = df.rename(columns={'S_n': 's_n', 'S_p': 's_p',
                            'PF_n': 'pf_n', 'PF_p': 'pf_p'})
    df = df.dropna()
    df['structure'] = df['structure'].map(ast.literal_eval)
    return df


def _preprocess_castelli_perovskites(df):
    df["formula"] = df["A"] + df["B"] + df["anion"]
    df['vbm'] = np.where(df['is_direct'], df['VB_dir'], df['VB_ind'])
    df['cbm'] = np.where(df['is_direct'], df['CB_dir'], df['CB_ind'])
    df['gap gllbsc'] = np.where(df['is_direct'], df['gllbsc_dir-gap'],
                                df['gllbsc_ind-gap'])
    df['structure'] = df['structure'].map(ast.literal_eval)
    dropcols = ["filename", "XCFunctional", "anion_idx", "Unnamed: 0", "A", "B",
                "anion", "gllbsc_ind-gap", "gllbsc_dir-gap", "CB_dir", "CB_ind",
                "VB_dir", "VB_ind"]
    df = df.drop(dropcols, axis=1)
    colmap = {"sum_magnetic_moments": "mu_b",
              "is_direct": "gap is direct",
              "heat_of_formation_all": "e_form",
              "FermiLevel": "fermi level",
              "FermiWidth": "fermi width"}
    df = df.rename(columns=colmap)
    df.reindex(sorted(df.columns), axis=1)
    return df


def _preprocess_elastic_tensor_2015(df):
    """
    Preprocessor used to convert the elastic_tensor_2015 dataset to the
    dataframe desired for JSON conversion

    Args:
        df (pandas.DataFrame)

    Returns: (pandas.DataFrame)
    """
    for i in list(df.index):
        for c in ['compliance_tensor', 'elastic_tensor',
                  'elastic_tensor_original']:
            df.at[(i, c)] = np.array(ast.literal_eval(df.at[(i, c)]))
    df['cif'] = df['structure']
    df['structure'] = pandas.Series([Poscar.from_string(s).structure
                                     for s in df['poscar']])
    new_columns = ['material_id', 'formula', 'nsites', 'space_group',
                   'volume', 'structure', 'elastic_anisotropy',
                   'G_Reuss', 'G_VRH', 'G_Voigt', 'K_Reuss', 'K_VRH',
                   'K_Voigt', 'poisson_ratio', 'compliance_tensor',
                   'elastic_tensor', 'elastic_tensor_original',
                   'cif', 'kpoint_density', 'poscar']
    return df[new_columns]


def _preprocess_piezoelectric_tensor(df):
    """
    Preprocessor used to convert the piezoelectric_constant dataset to the
    dataframe desired for JSON conversion

    Args:
        df (pandas.DataFrame)

    Returns: (pandas.DataFrame)
    """
    for i in list(df.index):
        c = 'piezoelectric_tensor'
        df.at[(i, c)] = np.array(ast.literal_eval(df.at[(i, c)]))
    df['cif'] = df['structure']
    df['structure'] = pandas.Series([Poscar.from_string(s).structure
                                     for s in df['poscar']])
    new_columns = ['material_id', 'formula', 'nsites', 'point_group',
                   'space_group', 'volume', 'structure', 'eij_max', 'v_max',
                   'piezoelectric_tensor', 'cif', 'meta', 'poscar']
    return df[new_columns]


def _preprocess_dielectric_constant(df):
    """
    Preprocessor used to convert the dielectric_constant dataset to the
    dataframe desired for JSON conversion

    Args:
        df (pandas.DataFrame)

    Returns: (pandas.DataFrame)
    """
    df['cif'] = df['structure']
    df['structure'] = pandas.Series([Poscar.from_string(s).structure
                                     for s in df['poscar']])
    new_columns = ['material_id', 'formula', 'nsites', 'space_group',
                   'volume', 'structure', 'band_gap', 'e_electronic',
                   'e_total', 'n', 'poly_electronic', 'poly_total',
                   'pot_ferroelectric', 'cif', 'meta', 'poscar']
    return df[new_columns]


def _preprocess_flla(df):
    """
    Preprocessor used to convert the flla dataset to the
    dataframe desired for JSON conversion

    Args:
        df (pandas.DataFrame)

    Returns: (pandas.DataFrame)
    """
    column_headers = ['material_id', 'e_above_hull', 'formula',
                      'nsites', 'structure', 'formation_energy',
                      'formation_energy_per_atom']
    df['structure'] = pandas.Series(
        [Structure.from_dict(ast.literal_eval(s))
         for s in df['structure']], df.index)
    return df[column_headers]


"""These dictionaries map the names of datasets to their preprocessors and to 
any special arguments that may be needed for their file to dataframe loader 
functions. Defaults to an identity function and no arguments"""


_datasets_to_preprocessing_routines = defaultdict(lambda x: x, {
    "elastic_tensor_2015": _preprocess_elastic_tensor_2015,
    "piezoelectric_tensor": _preprocess_piezoelectric_tensor,
    "dielectric_constant": _preprocess_dielectric_constant,
    "flla": _preprocess_flla,
    "castelli_perovskites": _preprocess_castelli_perovskites,
    "boltztrap_mp": _preprocess_boltztrap_mp,
})

_datasets_to_kwargs = defaultdict(dict, {
    "elastic_tensor_2015": {'comment': "#"},
    "piezoelectric_tensor": {'comment': "#"},
    "dielectric_constant": {'comment': "#"},
    "flla": {'comment': "#"},
    "boltztrap_mp": {'index_col': False},
})


def _file_to_dataframe(file_path, _dataset_name=None, preprocessing_func=None):
    """
    Converts dataset files to a dataframe using dataset specific predefined
    preprocessors or a preprocessing function passed as an argument.
    Returns the name of the dataset and a list of dataframes produced by the
    file preprocessing

    Args:
          file_path (str): file path to the dataset being processed to a
            dataframe

          _dataset_name (str): optional name of dataset, defaults to file name

          preprocessing_func (function): optional preprocessor

    Returns: (str, list of pandas.DataFrame)
    """

    # Default the dataset name to the file name if none provided
    if _dataset_name is None:
        _dataset_name = basename(file_path).split(".")[0]

    # Get keyword arguments for file reading functions
    loader_args = _datasets_to_kwargs[_dataset_name]

    # Read in the dataset from file
    if file_path.endswith(".csv"):
        df = pandas.read_csv(file_path, **loader_args)
    elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
        df = pandas.read_excel(file_path, **loader_args)
    elif file_path.endswith(".json"):
        df = pandas.read_json(file_path, **loader_args)
    else:
        raise ValueError("File type {} unsupported".format(file_path))

    # Apply a custom preprocessor if supplied, else do dictionary lookup
    # If dictionary lookup doesn't find a preprocessor none will be applied
    if preprocessing_func is None:
        if _dataset_name not in _datasets_to_preprocessing_routines.keys():
            raise UserWarning(
                "The dataset {} has no predefined preprocessor and will be "
                "loaded using only the default pandas.read_csv "
                "function.".format(_dataset_name)
            )
        df = _datasets_to_preprocessing_routines[_dataset_name](df)

    else:
        df = preprocessing_func(df)

    # Some preprocessors can return a list of dataframes, so make all returned
    # values be a list of dataframes
    if not isinstance(df, list):
        df = [df]

    return _dataset_name, df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This utility is for converting datasets to properly "
                    "formatted and encoded JSON files for storing dataframes "
                    "containing pymatgen based objects. It also creates a file "
                    "containing cryptographic hashes for the destination files."
                    " Currently supports csv, excel, and json file conversion."
    )
    parser.add_argument("-fp", "--file_paths", nargs="+", required=True,
                        help="File path to csv or file path to a directory "
                             "containing csv files, can list multiple sources")
    parser.add_argument("-d", "--destination",
                        help="Destination to place created JSON files")
    parser.add_argument("-mf", "--meta_file",
                        help="Optional path for metadata file")
    parser.add_argument("-nm", "--no_meta", action="store_true",
                        default=False, help="Flag to indicate no metadata")
    parser.add_argument("-ct", "--compression_type", choices=("gz", "bz2"))
    args = parser.parse_args()

    # Script supports multiple concurrent targets, so make a list of file paths
    # of each target file
    file_paths = []

    # Crawl directories for all files and add single files to path list
    for path in args.file_paths:
        if isdir(path):
            file_paths += [
                join(path, f) for f in listdir(path) if not isdir(path)
            ]
        else:
            file_paths.append(path)

    # Determine the destination to store results and ensure exists
    if args.destination is None:
        destination = expanduser(join("~", "dataset_to_json"))
    else:
        destination = args.destination

    makedirs(destination, exist_ok=True)

    # Set up the destination to store file metadata
    if args.meta_file is None and not args.no_meta:
        meta_file = join(destination, "file_meta.txt")
    else:
        meta_file = args.meta_file

    for f_path in file_paths:
        # Figure out the name of the dataset and
        # get a list of storage ready dataframes
        dataset_name, dataframe_list = _file_to_dataframe(f_path)
        # Store each dataframe and compute metadata if desired
        for num, dataframe in enumerate(dataframe_list):
            # Construct the file path to store dataframe at and store it
            df_num = ("_" + str(num) + "_") if len(dataframe_list) > 1 else ""
            json_destination = join(destination,
                                    dataset_name + df_num + ".json")
            store_dataframe_as_json(dataframe, json_destination,
                                    compression=args.compression_type)
            # Compute and store file metadata
            if not args.no_meta:
                with open(meta_file, "a") as out:
                    if args.compression_type is not None:
                        json_destination += ("." + args.compression_type)

                    file_hash = _get_file_sha256_hash(json_destination)

                    out.write(dataset_name + "\n" + file_hash + "\n")
                    out.write("\n")
