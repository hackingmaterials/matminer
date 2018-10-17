import ast
import argparse
from os import listdir, makedirs, sep
from os.path import isdir, join, expanduser, basename

import numpy as np
import pandas
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.core.structure import Structure

from matminer.datasets.utils import _get_file_sha256_hash
from matminer.utils.io import store_dataframe_as_json

__author__ = "Daniel Dopp <dbdopp@lbl.gov>"


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


_datasets_to_preprocessing_routines = {
    "elastic_tensor_2015": _preprocess_elastic_tensor_2015,
    "piezoelectric_tensor": _preprocess_piezoelectric_tensor,
    "dielectric_constant": _preprocess_dielectric_constant,
    "flla": _preprocess_flla,
    "castelli_perovskites": _preprocess_castelli_perovskites,
}


def _csv_to_dataframe(csv_path, _dataset_name=None, preprocessing_func=None):
    """
    Converts CSV files to a dataframe using dataset specific  predefined
    preprocessors or a preprocessing function passed as an argument

    Args:
          csv_path (str): file path to the csv being processed to a dataframe

          _dataset_name (str): optional name of dataset, defaults to file name

          preprocessing_func (function): optional preprocessor

    Returns: (pandas.DataFrame)
    """

    df = pandas.read_csv(csv_path, comment="#")

    if preprocessing_func is None:
        if _dataset_name is None:
            _dataset_name = csv_path.split(sep)[-1].split(".")[0]

        if _dataset_name not in _datasets_to_preprocessing_routines.keys():
            raise UserWarning(
                "The dataset you are trying to load has no "
                "predefined preprocessor and will be loaded "
                "using only the default pandas.read_csv function."
            )
        else:
            df = _datasets_to_preprocessing_routines[_dataset_name](df)
    else:
        df = preprocessing_func(df)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This utility is for converting datasets to properly "
                    "formatted and encoded JSON files for storing dataframes "
                    "containing pymatgen based objects. It also creates a file "
                    "containing cryptographic hashes for the destination files."
                    " Currently only supports csv datasets."
    )
    parser.add_argument("-fp", "--file_paths", nargs="+", required=True,
                        help="File path to csv or file path to a directory "
                             "containing csv files, can list multiple sources")
    parser.add_argument("-d", "--destination",
                        help="Destination to place created JSON files")
    parser.add_argument("-hf", "--hash_file",
                        help="Optional path for hash file")
    parser.add_argument("-nh", "--no_hashes", action="store_true",
                        default=False)
    parser.add_argument("-ct", "--compression_type", choices=("gz", "bz2"))
    args = parser.parse_args()

    csv_file_paths = []

    for path in args.file_paths:
        if isdir(path):
            csv_file_paths += [
                join(path, f) for f in listdir(path) if f.endswith(".csv")
            ]
        else:
            csv_file_paths.append(path)

    if args.destination is None:
        destination = expanduser(join("~", "csv_to_json"))
    else:
        destination = args.destination

    makedirs(destination, exist_ok=True)

    if args.hash_file is None:
        hash_file = join(destination, "file_hashes.txt")
    else:
        hash_file = args.hash_file

    if args.no_hashes:
        for file_path in csv_file_paths:
            dataframe = _csv_to_dataframe(file_path)
            store_dataframe_as_json(
                dataframe,
                join(destination, basename(file_path)[:-4] + ".json"),
                compression=args.compression_type
            )
    else:
        with open(hash_file, "w") as out:
            for file_path in csv_file_paths:
                dataset_name = basename(file_path)[:-4]
                json_destination = join(destination, dataset_name + ".json")

                dataframe = _csv_to_dataframe(file_path)

                store_dataframe_as_json(dataframe, json_destination,
                                        compression=args.compression_type)

                if args.compression_type is not None:
                    json_destination += ("." + args.compression_type)

                file_hash = _get_file_sha256_hash(json_destination)

                out.write(dataset_name + "\n" + file_hash + "\n\n")
