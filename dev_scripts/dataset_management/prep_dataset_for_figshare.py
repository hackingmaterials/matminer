import ast
import argparse
from time import sleep
from itertools import product
from math import inf
from os import listdir, makedirs
from os.path import isdir, join, expanduser, basename

import numpy as np
import pandas as pd
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition

from matminer.utils.io import store_dataframe_as_json, load_dataframe_from_json
from matminer.featurizers.conversions import StructureToComposition
from matminer.data_retrieval.retrieve_MP import MPDataRetrieval, MPRestError
from matminer.datasets.utils import _get_file_sha256_hash, \
    _read_dataframe_from_file

__author__ = "Daniel Dopp <dbdopp@lbl.gov>"

"""Each _preprocess_* function acts as a preprocessor for a dataset with a
given name. One should be written whenever a new dataset is being added to
matminer. These functions take the path of a given dataset and do the necessary
processing to make it a usable dataframe. If one dataframe is to be made from a
dataset, it should just return a name / dataframe pair, if more than one
dataframe is to be created a list of pairs should be returned."""


def _preprocess_tehrani_superhard_mat(file_path):
    df = _read_dataframe_from_file(file_path, header=None)

    # Strip space group from formula and also add compositions and update
    # formula to MP query compatible representation
    formula_list = []
    composition_list = []
    for formula, _ in [item.split(",") for item in df[0]]:
        comp = Composition(formula)

        formula_list.append(comp.get_reduced_formula_and_factor()[0])
        composition_list.append(comp)

    df[0] = formula_list
    df["composition"] = composition_list

    # Give columns descriptive names
    column_map = {0: "formula", 1: "bulk_modulus", 2: "shear_modulus"}

    composition_features = ["atomic_number", "atomic_weight", "period_number",
                            "group_number", "family_number", "Mendeleev_number",
                            "atomic_radius", "covalent_radius",
                            "Zungar_radius", "ionic_radius", "crystal_radius",
                            "Pauling_EN", "Martynov_EN", "Gordy_EN",
                            "Mulliken_EN", "Allred-Rochow_EN",
                            "metallic_valence", "number_VE",
                            "Gillman_number_VE", "number_s_electrons",
                            "number_p_electrons", "number_d_electrons",
                            "number_outer_shell_electrons",
                            "first_ionization_energy", "polarizability",
                            "melting_point", "boiling_point", "density",
                            "specific_heat", "heat_of_fusion",
                            "heat_of_vaporization", "thermal_conductivity",
                            "heat_of_atomization", "cohesive_energy"]

    for feature_with_variant in product(composition_features,
                                        ["feat_1", "feat_2",
                                         "feat_3", "feat_4"]):
        column_map[len(column_map)] = "_".join(feature_with_variant)

    structure_features = ["space_group_number", "crystal_system", "Laue_class",
                          "crystal_class", "inversion_centre", "polar_axis",
                          "reduced_volume", "density", "anisotropy",
                          "electron_density", "volume_per_atom",
                          "valence_electron_density", "Gilman_electron_density",
                          "outer_shell_electron_density"]

    for feature in structure_features:
        column_map[len(column_map)] = feature

    df.rename(columns=column_map, inplace=True)

    # Add columns for data we don't have yet
    props_to_save = ['material_id', 'initial_structure', 'structure']
    df = df.reindex(index=df.index, columns=df.columns.tolist() + props_to_save)

    # Query Materials Project to get structure and mpid data for each entry
    num_mismatching_entries = 0
    num_missing_entries = 0
    num_malformed_entries = 0
    mp_retriever = MPDataRetrieval()
    props_to_query = props_to_save + ["elasticity.K_VRH", "elasticity.G_VRH"]

    for material_index, material in df.iterrows():
        if material_index % 10 == 0:
            print("Processing material {}".format(material_index))

        retriever_criteria = {
            'pretty_formula': material['formula'],
            'spacegroup.number': material['space_group_number']
        }

        # While loop with try except block to handle server response errors
        while True:
            try:
                mp_query = mp_retriever.get_dataframe(
                    criteria=retriever_criteria,
                    properties=props_to_query,
                    index_mpid=False
                )

                # Query was successful, exit loop
                break

            except MPRestError:
                print("Server returned error, waiting 10 seconds then retrying")
                sleep(10)

        # Throw out entry if no hits on MP
        if len(mp_query) == 0:
            print(
                "No materials found for entry with formula {} and "
                "space group {}".format(material['formula'],
                                        material['space_group_number'])
            )
            print("Data point will be thrown out\n")
            df.drop(material_index, inplace=True)
            num_missing_entries += 1

        else:
            closest_mat_index = 0
            # If more than one material id is found, select the closest one
            if len(mp_query) > 1:
                closest_mat_distance = inf
                # For each mat calculate euclidean distance from db entry
                for i, entry in mp_query.iterrows():
                    bulk_dif = (entry["elasticity.K_VRH"]
                                - material["bulk_modulus"])
                    shear_dif = (entry["elasticity.G_VRH"]
                                 - material["shear_modulus"])
                    mat_dist = (bulk_dif ** 2 + shear_dif ** 2) ** .5

                    # Select the material that is closest in space to db entry
                    if mat_dist < closest_mat_distance:
                        closest_mat_index = i
                        closest_mat_distance = mat_dist

            mat_data = mp_query.loc[closest_mat_index]

            # Check to ensure selected entry doesn't lack any data
            if np.any(mat_data.isna()):
                print(
                    "Malformed data encountered for entry with formula {} and "
                    "space group {}".format(material['formula'],
                                            material['space_group_number'])
                )
                print("Data point will be thrown out\n")
                df.drop(material_index, inplace=True)
                num_malformed_entries += 1

            else:
                # Check similarity of  dataset entry to the selected MP entry
                bulk_abs_dif = abs(
                    mat_data["elasticity.K_VRH"] - material["bulk_modulus"]
                )
                shear_abs_dif = abs(
                    mat_data["elasticity.G_VRH"] - material["shear_modulus"]
                )

                bulk_relative_dif = bulk_abs_dif / material["bulk_modulus"]
                shear_relative_dif = shear_abs_dif / material["shear_modulus"]

                # Discard entry if MP entry is too different from dataset entry
                if ((bulk_relative_dif > .05 or shear_relative_dif > .05)
                        and (bulk_abs_dif > 1 or shear_abs_dif > 1)):
                    print("Warning: MP entry selected for {} with space group "
                          "{} has a difference in elastic data greater than "
                          "5 percent/1GPa!".format(material["formula"],
                                                   material[
                                                       "space_group_number"
                                                   ]))
                    print("Bulk moduli dataset vs. MP:",
                          material["bulk_modulus"],
                          mat_data["elasticity.K_VRH"])
                    print("Shear moduli dataset vs. MP:",
                          material["shear_modulus"],
                          mat_data["elasticity.G_VRH"])
                    print("Data point will be thrown out\n")
                    df.drop(material_index, inplace=True)
                    num_mismatching_entries += 1

                else:
                    df.loc[material_index, props_to_save] = mat_data[
                        props_to_save]

    # Report on discarded entries
    if num_mismatching_entries:
        print("{} entries discarded for value mismatch".format(
            num_mismatching_entries
        ))

    if num_missing_entries:
        print("{} entries discarded for no hit on MP".format(
            num_missing_entries
        ))

    if num_malformed_entries:
        print("{} entries discarded for malformed data".format(
            num_malformed_entries
        ))

    # Turn structure strings into Structure objects
    for alias in ['structure', 'initial_structure']:
        df[alias] = pd.Series([Structure.from_dict(s)
                               for s in df[alias]], df.index)

    return "tehrani_superhard_mat", df


def _preprocess_wolverton_oxides(file_path):
    df = _read_dataframe_from_file(file_path)

    dropcols = ['In literature', 'Valence A', 'Valence B', 'Radius A [ang]',
                'Radius B [ang]']
    df = df.drop(dropcols, axis=1)

    colmap = {"Chemical formula": "formula",
              "A": "atom a",
              "B": "atom b",
              "Formation energy [eV/atom]": "e_form",
              "Band gap [eV]": "gap pbe",
              "Magnetic moment [mu_B]": "mu_b",
              "Vacancy energy [eV/O atom]": "e_form oxygen",
              "Stability [eV/atom]": "e_hull",
              "Volume per atom [A^3/atom]": 'vpa',
              "a [ang]": "a",
              "b [ang]": "b",
              "c [ang]": "c",
              "alpha [deg]": "alpha",
              "beta [deg]": "beta",
              "gamma [deg]": "gamma",
              "Lowest distortion": "lowest distortion"}
    df = df.rename(columns=colmap)

    for k in ['e_form', 'gap pbe', 'e_hull', 'vpa', 'e_form oxygen']:
        df[k] = pd.to_numeric(df[k], errors='coerce')

    return "wolverton_oxides", df.dropna()


def _preprocess_m2ax(file_path):
    df = _read_dataframe_from_file(file_path)

    colmap = {"M2AXphase": "formula",
              "B": "bulk modulus",
              "G": "shear modulus",
              "E": "elastic modulus",
              "C11": "c11",
              "C12": "c12",
              "C13": "c13",
              "C33": "c33",
              "C44": "c44",
              "dMX": "d_mx",
              "dMA": "d_ma"}

    return "m2ax", df.rename(columns=colmap)


def _preprocess_glass_binary(file_path):
    df = _read_dataframe_from_file(file_path)

    return "glass_binary", df


def _preprocess_expt_formation_enthalpy(file_path):
    df = _read_dataframe_from_file(file_path)

    return "expt_formation_enthalpy", df


def _preprocess_expt_gap(file_path):
    df = _read_dataframe_from_file(file_path)

    df = df.rename(columns={'composition': 'formula', 'Eg (eV)': 'gap expt'})
    # The numbers in 323 formulas such as 'AgCNO,65' or 'Sr2MgReO6,225' are
    # space group numbers confirmed by Jakoah Brgoch the corresponding author
    df['formula'] = df['formula'].apply(lambda x: x.split(',')[0])

    return "expt_gap", df


def _preprocess_jarvis_dft_2d(file_path):
    df = load_dataframe_from_json(file_path)

    colmap = {
        "epsx": "epsilon_x opt",
        "epsy": "epsilon_y opt",
        "epsz": "epsilon_z opt",
        'final_str': 'structure',
        'initial_str': 'structure initial',
        'form_enp': 'e_form',
        "mbj_gap": "gap tbmbj",
        "mepsx": "epsilon_x tbmbj",
        "mepsy": "epsilon_y tbmbj",
        "mepsz": "epsilon_z tbmbj",
        "op_gap": "gap opt",
    }
    df = df.rename(columns=colmap)

    dropcols = ['elastic', 'fin_en', 'magmom', 'kpoints', 'incar', 'phi']
    df = df.drop(dropcols, axis=1)

    s = StructureToComposition()
    df = s.featurize_dataframe(df, "structure")

    for k in ["e_form", "epsilon_x opt", "epsilon_y opt", "epsilon_z opt",
              "gap tbmbj", "epsilon_x tbmbj", "epsilon_y tbmbj",
              "epsilon_z tbmbj", "gap opt"]:
        df[k] = pd.to_numeric(df[k], errors='coerce')

    return "jarvis_dft_2d", df


def _preprocess_jarvis_dft_3d(file_path):
    df = load_dataframe_from_json(file_path)

    colmap = {
        "epsx": "epsilon_x opt",
        "epsy": "epsilon_y opt",
        "epsz": "epsilon_z opt",
        "final_str": "structure",
        "form_enp": "e_form",
        "gv": "shear modulus",
        'initial_str': 'structure initial',
        "kv": "bulk modulus",
        "mbj_gap": "gap tbmbj",
        "mepsx": "epsilon_x tbmbj",
        "mepsy": "epsilon_y tbmbj",
        "mepsz": "epsilon_z tbmbj",
        "op_gap": "gap opt",
    }

    df = df.rename(columns=colmap)
    # Remove redundant or unneeded for our purposes columns
    df = df.drop(["eff_mass", "elastic", "encut", "fin_en", "icsd", "incar",
                  "kp_leng", "kpoints", "magmom", ], axis=1)

    s = StructureToComposition()
    df = s.featurize_dataframe(df, "structure")

    for k in ["e_form", "epsilon_x opt", "epsilon_y opt", "epsilon_z opt",
              "shear modulus", "bulk modulus", "gap tbmbj", "epsilon_x tbmbj",
              "epsilon_y tbmbj", "epsilon_z tbmbj", "gap opt"]:
        df[k] = pd.to_numeric(df[k], errors='coerce')

    return "jarvis_dft_3d", df


def _preprocess_heusler_magnetic(file_path):
    df = _read_dataframe_from_file(file_path)

    dropcols = ['gap width', 'stability']
    df = df.drop(dropcols, axis=1)

    return "heusler_magnetic", df


def _preprocess_steel_strength(file_path):
    df = _read_dataframe_from_file(file_path)

    return "steel_strength", df


def _preprocess_glass_ternary_hipt(file_path):
    df = _read_dataframe_from_file(file_path)

    return "glass_ternary_hipt", df


def _preprocess_jarvis_ml_dft_training(file_path):
    df = load_dataframe_from_json(file_path)

    colmap = {
        "el_mass_x": "e mass_x",
        "el_mass_y": "e mass_y",
        "el_mass_z": "e mass_z",
        "epsx": "epsilon_x opt",
        "epsy": "epsilon_y opt",
        "epsz": "epsilon_z opt",
        "exfoliation_en": "e_exfol",
        "form_enp": "e_form",
        "gv": "shear modulus",
        "hl_mass_x": "hole mass_x",
        "hl_mass_y": "hole mass_y",
        "hl_mass_z": "hole mass_z",
        "kv": "bulk modulus",
        "magmom": "mu_b",
        "mbj_gap": "gap tbmbj",
        "mepsx": "epsilon_x tbmbj",
        "mepsy": "epsilon_y tbmbj",
        "mepsz": "epsilon_z tbmbj",
        "op_gap": "gap opt",
        "strt": "structure",
    }

    df = df.rename(columns=colmap)
    # Remove redundant or unneeded for our purposes columns
    df = df.drop(["desc", "el_mass", "encut", "fin_enp", "hl_mass",
                  "kp_leng", "multi_elastic", "type"], axis=1)
    s = StructureToComposition()
    df = s.featurize_dataframe(df, "structure")

    for k in ["e mass_x", "e mass_y", "e mass_z", "e_exfol", "e_form", "mu_b",
              "epsilon_x opt", "epsilon_y opt", "epsilon_z opt",
              "shear modulus", "hole mass_x", "hole mass_y", "hole mass_z",
              "bulk modulus", "gap tbmbj", "epsilon_x tbmbj", "epsilon_y tbmbj",
              "epsilon_z tbmbj", "gap opt"]:
        df[k] = pd.to_numeric(df[k], errors='coerce')

    return "jarvis_ml_dft_training", df


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


"""These dictionaries map the filename of datasets to their preprocessors.
Defaults to just loading in the file with default pd load function for a
given file type"""


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
    "wolverton_oxides": _preprocess_wolverton_oxides,
    "m2ax_elastic": _preprocess_m2ax,
    "glass_binary": _preprocess_glass_binary,
    "formation_enthalpy_expt": _preprocess_expt_formation_enthalpy,
    "zhuo_gap_expt": _preprocess_expt_gap,
    "jdft_2d-7-7-2018": _preprocess_jarvis_dft_2d,
    "heusler_magnetic": _preprocess_heusler_magnetic,
    "steel_strength": _preprocess_steel_strength,
    "jarvisml_cfid": _preprocess_jarvis_ml_dft_training,
    "glass_ternary_hipt": _preprocess_glass_ternary_hipt,
    "jdft_3d-7-7-2018": _preprocess_jarvis_dft_3d,
    "tehrani_superhard_mat": _preprocess_tehrani_superhard_mat,
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
                join(path, f) for f in listdir(path) if not isdir(f)
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
        for index, dataframe in enumerate(dataframe_list):
            # Construct the file path to store dataframe at and store it
            # Str conversion purely to get rid of an annoying type warning
            json_destination = join(destination,
                                    str(dataset_names[index]) + ".json")
            store_dataframe_as_json(dataframe, json_destination,
                                    compression=args.compression_type)
            # Compute and store file metadata
            if not args.no_meta:
                with open(meta_file, "a") as out:
                    if args.compression_type is not None:
                        json_destination += ("." + args.compression_type)

                    file_hash = _get_file_sha256_hash(json_destination)

                    out.write(str(dataset_names[index])
                              + "\nhash: "
                              + file_hash
                              + "\n")
                    out.write("column types:\n")
                    out.write(dataframe.dtypes.to_string())
                    out.write("\n")

                    out.write("num_entries:\n")
                    out.write(str(len(dataframe)))

                    out.write("\n\n")
