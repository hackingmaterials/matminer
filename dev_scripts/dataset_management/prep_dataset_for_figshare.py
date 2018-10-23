import ast
import argparse
from os import listdir, makedirs
from os.path import isdir, join, expanduser, basename

import numpy as np
import pandas as pd
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.core.structure import Structure

from matminer.datasets.utils import _get_file_sha256_hash, \
    _read_dataframe_from_file
from matminer.utils.io import store_dataframe_as_json

__author__ = "Daniel Dopp <dbdopp@lbl.gov>"

"""
Each _preprocess_* function acts as a preprocessor for a dataset with a
given name. One should be written whenever a new dataset is being added to
matminer. These functions take the path of a given dataset and do the necessary
processing to make it a usable dataframe. If one dataframe is to be made from a 
dataset, it should just return a name / dataframe pair, if more than one 
dataframe is to be created a list of pairs should be returned. 
"""


def _preprocess_mp(file_path):
    df = _read_dataframe_from_file(file_path)

    dropcols = ['energy', 'energy_per_atom']
    df = df.drop(dropcols, axis=1)
    for alias in ['structure', 'initial_structure']:
        if alias in df.columns.values:
            df[alias] = df[alias].map(ast.literal_eval)
    colmap = {'material_id': 'mpid',
              'pretty_formula': 'formula',
              'band_gap': 'gap pbe',
              'e_above_hull': 'e_hull',
              'elasticity.K_VRH': 'bulk modulus',
              'elasticity.G_VRH': 'shear modulus',
              'elasticity.elastic_anisotropy': 'elastic anisotropy',
              'total_magnetization': 'mu_b',
              'initial_structure': 'initial structure',
              'formation_energy_per_atom': 'e_form'}
    if 'structure' in df.columns or 'initial_structure' in df.columns:
        dataname = "mp_all"
    else:
        dataname = "mp_nostruct"

    return dataname, df.rename(columns=colmap)


def _preprocess_glass_ternary_landolt(file_path):
    df = _read_dataframe_from_file(file_path)

    df.drop_duplicates()
    df["gfa"] = df["phase"].apply(lambda x: 1 if x == "AM" else 0)

    return "glass_ternary_landolt", df


def _preprocess_citrine_thermal_conductivity(file_path):
    df = _read_dataframe_from_file(file_path)

    df = df[df['k-units'].isin(
        ['W/m.K', 'W/m$\\cdot$K', 'W/mK', 'W\\m K', 'Wm$^{-1}$K$^{-1}$'])]

    return "citrine_thermal_conductivity", df


def _preprocess_double_perovskites_gap(file_path):
    df = pd.read_excel(file_path, sheet_name='bandgap')

    df = df.rename(columns={'A1_atom': 'a_1', 'B1_atom': 'b_1',
                            'A2_atom': 'a_2', 'B2_atom': 'b_2'})
    lumo = pd.read_excel(file_path, sheet_name='lumo')

    return ["double_perovskites_gap", "double_perovskites_gap_lumo"], [df, lumo]


def _preprocess_phonon_dielectric_mp(file_path):
    df = _read_dataframe_from_file(file_path)

    df = df[df['asr_breaking'] < 30].drop('asr_breaking', axis=1)
    # remove entries not having structure, formula, or a target
    df = df.dropna()
    df['structure'] = df['structure'].map(ast.literal_eval)

    return 'phonon_dielectric_mp', df.reset_index(drop=True)


def _preprocess_boltztrap_mp(file_path):
    df = _read_dataframe_from_file(file_path, index_col=False)

    df = df.rename(columns={'S_n': 's_n', 'S_p': 's_p',
                            'PF_n': 'pf_n', 'PF_p': 'pf_p'})
    df = df.dropna()
    df['structure'] = df['structure'].map(ast.literal_eval)

    return 'boltztrap_mp', df


def _preprocess_castelli_perovskites(file_path):
    df = _read_dataframe_from_file(file_path)

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

    return 'castelli_perovskites', df


def _preprocess_elastic_tensor_2015(file_path):
    """
    Preprocessor used to convert the elastic_tensor_2015 dataset to the
    dataframe desired for JSON conversion

    Args:
        file_path (str)

    Returns: (pd.DataFrame)
    """
    df = _read_dataframe_from_file(file_path, comment="#")

    for i in list(df.index):
        for c in ['compliance_tensor', 'elastic_tensor',
                  'elastic_tensor_original']:
            df.at[(i, c)] = np.array(ast.literal_eval(df.at[(i, c)]))
    df['cif'] = df['structure']
    df['structure'] = pd.Series([Poscar.from_string(s).structure
                                     for s in df['poscar']])
    new_columns = ['material_id', 'formula', 'nsites', 'space_group',
                   'volume', 'structure', 'elastic_anisotropy',
                   'G_Reuss', 'G_VRH', 'G_Voigt', 'K_Reuss', 'K_VRH',
                   'K_Voigt', 'poisson_ratio', 'compliance_tensor',
                   'elastic_tensor', 'elastic_tensor_original',
                   'cif', 'kpoint_density', 'poscar']

    return 'elastic_tensor_2015', df[new_columns]


def _preprocess_piezoelectric_tensor(file_path):
    """
    Preprocessor used to convert the piezoelectric_constant dataset to the
    dataframe desired for JSON conversion

    Args:
        file_path (str)

    Returns: (pd.DataFrame)
    """
    df = _read_dataframe_from_file(file_path, comment="#")

    for i in list(df.index):
        c = 'piezoelectric_tensor'
        df.at[(i, c)] = np.array(ast.literal_eval(df.at[(i, c)]))
    df['cif'] = df['structure']
    df['structure'] = pd.Series([Poscar.from_string(s).structure
                                     for s in df['poscar']])
    new_columns = ['material_id', 'formula', 'nsites', 'point_group',
                   'space_group', 'volume', 'structure', 'eij_max', 'v_max',
                   'piezoelectric_tensor', 'cif', 'meta', 'poscar']

    return 'piezoelectric_tensor', df[new_columns]


def _preprocess_dielectric_constant(file_path):
    """
    Preprocessor used to convert the dielectric_constant dataset to the
    dataframe desired for JSON conversion

    Args:
        file_path (str)

    Returns: (pd.DataFrame)
    """
    df = _read_dataframe_from_file(file_path, comment="#")

    df['cif'] = df['structure']
    df['structure'] = pd.Series([Poscar.from_string(s).structure
                                     for s in df['poscar']])
    new_columns = ['material_id', 'formula', 'nsites', 'space_group',
                   'volume', 'structure', 'band_gap', 'e_electronic',
                   'e_total', 'n', 'poly_electronic', 'poly_total',
                   'pot_ferroelectric', 'cif', 'meta', 'poscar']

    return 'dielectric_constant', df[new_columns]


def _preprocess_flla(file_path):
    """
    Preprocessor used to convert the flla dataset to the
    dataframe desired for JSON conversion

    Args:
        file_path (str)

    Returns: (pd.DataFrame)
    """
    df = _read_dataframe_from_file(file_path, comment="#")

    column_headers = ['material_id', 'e_above_hull', 'formula',
                      'nsites', 'structure', 'formation_energy',
                      'formation_energy_per_atom']
    df['structure'] = pd.Series(
        [Structure.from_dict(ast.literal_eval(s))
         for s in df['structure']], df.index)

    return 'flla', df[column_headers]


"""
These dictionaries map the filename of datasets to their preprocessors. 
Defaults to just loading in the file with default pd load function for a 
given file type
"""


_datasets_to_preprocessing_routines = {
    "elastic_tensor_2015": _preprocess_elastic_tensor_2015,
    "piezoelectric_tensor": _preprocess_piezoelectric_tensor,
    "dielectric_constant": _preprocess_dielectric_constant,
    "flla": _preprocess_flla,
    "castelli_perovskites": _preprocess_castelli_perovskites,
    "boltztrap_mp": _preprocess_boltztrap_mp,
    "phonon_dielectric_mp": _preprocess_phonon_dielectric_mp,
    "double_perovskites_gap": _preprocess_double_perovskites_gap,
    "citrine_thermal_conductivity": _preprocess_citrine_thermal_conductivity,
    "glass_ternary_landolt": _preprocess_glass_ternary_landolt,
    "mp_all": _preprocess_mp,
    "mp_nostruct": _preprocess_mp,
}


def _file_to_dataframe(file_path):
    """
    Converts dataset files to a dataframe(s) using dataset specific predefined
    preprocessors or just the standard pandas load* function if unrecognized.
    Returns the names of the datasets produced and a list of dataframes
    produced by the file preprocessing

    Args:
          file_path (str): file path to the dataset being processed to a
            dataframe

    Returns: (list of str, list of pd.DataFrame)
    """

    file_name = basename(file_path).split(".")[0]

    # Apply a custom preprocessor if supplied, else do dictionary lookup
    # If dictionary lookup doesn't find a preprocessor none will be applied
    if file_name not in _datasets_to_preprocessing_routines.keys():
        print(
            "Warning: The dataset {} has no predefined preprocessor "
            "and will be loaded using only the default pd.read_* "
            "function.".format(file_name), flush=True
        )
        df = _read_dataframe_from_file(file_path)

    else:
        file_name, df = _datasets_to_preprocessing_routines[
            file_name](file_path)

    # Some preprocessors can return a list of dataframes,
    # so make all returned values be lists for consistency
    if not isinstance(df, list):
        df = [df]
    if not isinstance(file_name, list):
        file_name = [file_name]

    return file_name, df


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
        dataset_names, dataframe_list = _file_to_dataframe(f_path)
        # Store each dataframe and compute metadata if desired
        for i in range(len(dataframe_list)):
            # Construct the file path to store dataframe at and store it
            # Str conversion purely to get rid of an annoying type warning
            json_destination = join(destination,
                                    str(dataset_names[i]) + ".json")
            store_dataframe_as_json(dataframe_list[i], json_destination,
                                    compression=args.compression_type)
            # Compute and store file metadata
            if not args.no_meta:
                with open(meta_file, "a") as out:
                    if args.compression_type is not None:
                        json_destination += ("." + args.compression_type)

                    file_hash = _get_file_sha256_hash(json_destination)

                    out.write(str(dataset_names[i])
                              + "\nhash: "
                              + file_hash
                              + "\n")
                    out.write("column types:\n")
                    out.write(dataframe_list[i].dtypes.to_string())
                    out.write("\n")

                    out.write("num_entries:\n")
                    out.write(str(len(dataframe_list[i])))

                    out.write("\n\n")
