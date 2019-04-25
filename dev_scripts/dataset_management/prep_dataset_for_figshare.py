import logging
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
from tqdm import tqdm

from matminer.utils.io import store_dataframe_as_json, load_dataframe_from_json
from matminer.featurizers.conversions import StructureToComposition
from matminer.data_retrieval.retrieve_MP import MPDataRetrieval, MPRestError
from matminer.datasets.utils import _get_file_sha256_hash, \
    _read_dataframe_from_file

__author__ = "Daniel Dopp <danieldopp@outlook.com>"

logging.basicConfig(filename='dataset_prep.log', level='INFO')

"""Each _preprocess_* function acts as a preprocessor for a dataset with a
given name. One should be written whenever a new dataset is being added to
matminer. These functions take the path of a given dataset and do the necessary
processing to make it a usable dataframe. If one dataframe is to be made from a
dataset, it should just return a name / dataframe pair, if more than one
dataframe is to be created a list of pairs should be returned.

All preprocessing functions return the data in pandas.DataFrame with a
dataset name string. In each method a raw data file is loaded,
some preliminary transformation/renaming/cleaning done and
the result (name, df) is returned.

If you plan to add a new dataset please follow the guidelines and refer to
documentation in the matminer docs for consistency. Generally, using
column names and unit conventions already used in other preprocessing functions
is preferred (e.g., always using e_form for heat of formation in eV).

(Loose) Naming convention guidelines:
    - use small letters for column names consistently
    - return only those columns that cannot be derived from other available ones
    - use spaces between words; use _ only when it makes sense as a subscript
        e.g. "e_form" means energy of formation
    - start with property name followed by method/additional description:
        e.g. "gap expt" means band gap measured experimentally
        e.g. "gap pbe" means band gap calculated via DFT using PBE functional
    - avoid including units in the column name, instead explain in the docs
    - roughly use a 15-character limit for column names

Data convention guidelines
    - If structures/composition are present, the dataframe should have
    them contained in a column where each entry is a pymatgen
    Structure/Composition object
    - The structures should NOT be strings (MP queries can return strings via
        REST, so be cautious)
    - To convert strings to dictionary, use ast.literal_eval
    - TO convert string to pymatgen object, use
    pymatgen.core.structure/composition"""


def _clear_incomplete_dataframe_rows(df, col_names=None):
    if col_names is None:
        col_names = df.columns

    mask = np.any(df[col_names].isna(), axis=1)

    return df[~mask]


def _preprocess_brgoch_superhard_training(file_path):
    """
    2574 materials used for training shear and bulk modulus predictors.

    References:
        https://pubs.acs.org/doi/pdf/10.1021/jacs.8b02717

     Returns:
            formula (str): Chemical formula
            bulk_modulus (float): VRH bulk modulus
            shear_modulus (float): VRH shear modulus
            composition (Composition): pymatgen composition object
            material_id (str): materials project id
            structure (Structure): pymatgen structure object
            brgoch_feats (dict): features used in brgoch study
                                 (see dataset reference)
            suspect_value (bool): True if bulk/shear value doesn't closely match
                                  MP data as of dataset generation

    """
    logging.info("Processing brgoch_superhard_training dataset")
    # header=None because original xlsx had no column labels
    df = _read_dataframe_from_file(file_path, header=None)

    # Strip space group from formula, add compositions, update
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

    composition_features = []
    composition_base_names = [
        "atomic_number", "atomic_weight", "period_number",
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
        "heat_of_atomization", "cohesive_energy"
    ]

    for feature_with_variant in product(composition_base_names,
                                        ["feat_1", "feat_2",
                                         "feat_3", "feat_4"]):
        feat_name = "_".join(feature_with_variant)
        column_map[len(column_map)] = feat_name
        composition_features.append(feat_name)

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
    props_to_save = ['material_id', 'structure']
    df = df.reindex(index=df.index, columns=(df.columns.tolist()
                                             + props_to_save
                                             + ["brgoch_feats",
                                                "suspect_value"]))
    df["brgoch_feats"] = [{} for _ in range(len(df))]
    df["suspect_value"] = [False for _ in range(len(df))]

    # Query Materials Project to get structure and mpid data for each entry
    mp_retriever = MPDataRetrieval()
    props_to_query = props_to_save + ["elasticity.K_VRH", "elasticity.G_VRH"]

    for material_index in tqdm(range(len(df)), desc="Processing Dataset"):
        material = df.loc[material_index]

        # Zip features from original paper into single column
        brgoch_feats = df.loc[
            material_index, composition_features + structure_features
        ]
        df.loc[material_index, "brgoch_feats"].update(brgoch_feats.to_dict())

        retriever_criteria = {
            'pretty_formula': str(material['formula']),
            'spacegroup.number': int(material['space_group_number'])
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
                logging.warning("MP query failed, sleeping for 3 seconds")
                sleep(3)

        # Clean retrieved query, removes entries with missing elasticity data
        mp_query = _clear_incomplete_dataframe_rows(mp_query)

        # Throw out entry if no hits on MP
        if len(mp_query) == 0:
            error_msg = "\nNo valid materials found for entry with formula " \
                        "{} and space group {}. Data point will be marked as " \
                        "suspect\n".format(material['formula'],
                                           material['space_group_number'])
            logging.warning(error_msg)
            df.loc[material_index, "suspect_value"] = True

        else:
            closest_mat_index = min(mp_query.index)
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

            # Mark suspect if MP entry is too different from dataset entry
            bulk_abs_dif = abs(
                mat_data["elasticity.K_VRH"] - material["bulk_modulus"]
            )
            shear_abs_dif = abs(
                mat_data["elasticity.G_VRH"] - material["shear_modulus"]
            )

            bulk_relative_dif = bulk_abs_dif / material["bulk_modulus"]
            shear_relative_dif = shear_abs_dif / material["shear_modulus"]

            if ((bulk_relative_dif > .05 and bulk_abs_dif > 1)
                    or (shear_relative_dif > .05 and shear_abs_dif > 1)):
                err_msg = "\nMP entry selected for {} with space group {} " \
                          "has a difference in elastic data greater than 5 " \
                          "percent/1GPa!".format(material["formula"],
                                                 material["space_group_number"])

                err_msg += "\nBulk moduli dataset vs. MP: {} {}".format(
                    material["bulk_modulus"],
                    mat_data["elasticity.K_VRH"]
                )
                err_msg += "\nShear moduli dataset vs. MP: {} {}".format(
                    material["shear_modulus"],
                    mat_data["elasticity.G_VRH"]
                )
                err_msg += "\nData point will be marked as suspect\n"

                logging.warning(err_msg)
                df.loc[material_index, "suspect_value"] = True

            df.loc[material_index, props_to_save] = mat_data[props_to_save]

    # Drop brgoch features from main columns since now zipped under brgoch_feats
    df.drop(labels=composition_features + structure_features,
            axis=1, inplace=True)

    # Report on discarded entries
    if np.any(df["suspect_value"]):
        print("{} entries could not be accurately cross referenced with "
              "Materials Project. "
              "See log file for details".format(len(df[df["suspect_value"]])))

    # Turn structure dicts into Structure objects,
    # leave missing structures as nan values
    structure_obs = []
    for s in df["structure"]:
        if not isinstance(s, Structure):
            if isinstance(s, dict):
                s = Structure.from_dict(s)
            elif not isinstance(s, np.float) and np.isnan(s):
                raise ValueError("Something went wrong, invalid "
                                 "value {} in structure column".format(s))
        structure_obs.append(s)

    df["structure"] = structure_obs

    return "brgoch_superhard_training", df


def _preprocess_wolverton_oxides(file_path):
    """
        4914 perovskite oxides containing composition data, lattice constants,
        and formation + vacancy formation energies. All perovskites are of the
        form ABO3.

        References:
            https://www.nature.com/articles/sdata2017153#ref40

        Returns:
            formula (input): Chemical formula
            atom a (input): The atom in the 'A' site of the pervoskite.
            atom b (input): The atom in the 'B' site of the perovskite.
            a (input): Lattice parameter a, in A (angstrom)
            b (input): Lattice parameter b, in A
            c (input): Lattice parameter c, in A
            alpha (input): Lattice angle alpha, in degrees
            beta (input): Lattice angle beta, in degrees
            gamma (input): Lattice angle gamma, in degrees

            lowest distortion (target): Local distortion crystal structure with
                lowest energy among all considered distortions.
            e_form (target): Formation energy in eV
            gap pbe (target): Bandgap in eV from PBE calculations
            mu_b (target): Magnetic moment
            e_form oxygen (target): Formation energy of oxygen vacancy (eV)
            e_hull (target): Energy above convex hull, wrt. OQMD db (eV)
            vpa (target): Volume per atom (A^3/atom)
        """
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
    """
    Elastic properties of 223 stable M2AX compounds from "A comprehensive survey
    of M2AX phase elastic properties" by Cover et al. Calculations are PAW
    PW91.

    References:
        http://iopscience.iop.org/article/10.1088/0953-8984/21/30/305403/meta

    Returns:
        formula (input):
        a (input): Lattice parameter a, in A (angstrom)
        c (input): Lattice parameter c, in A
        d_mx (input): Distance from the M atom to the X atom
        d_ma (input): Distance from the M atom to the A atom

        c11/c12/c13/c33/c44 (target): Elastic constants of the M2AX material.
            These are specific to hexagonal materials.
        bulk modulus (target): in GPa
        shear modulus (target): in GPa
        elastic modulus (target): in GPa
    """
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
    """
    Metallic glass formation data for binary alloys, collected from various
    experimental techniques such as melt-spinning or mechanical alloying.
    This dataset covers all compositions with an interval of 5 at.% in 59
    binary systems, containing a total of 5959 alloys in the dataset.
    The target property of this dataset is the glass forming ability (GFA),
    i.e. whether the composition can form monolithic glass or not, which is
    either 1 for glass forming or 0 for non-full glass forming.

    References:
        https://pubs.acs.org/doi/10.1021/acs.jpclett.7b01046

    Returns:
        formula (input): chemical formula
        phase (target): only in the "ternary" dataset, designating the phase
                        obtained in glass producing experiments,
                        "AM": amorphous phase
                        "CR": crystalline phase
        gfa (target): glass forming ability, correlated with the phase column,
                      designating whether the composition can form monolithic
                      glass or not,
                      1: glass forming ("AM")
                      0: non-full-glass forming ("CR")

    """
    df = _read_dataframe_from_file(file_path)

    return "glass_binary", df


def _preprocess_glass_binary_v2(file_path):
    """
    Metallic glass formation data for binary alloys, collected from various
    experimental techniques such as melt-spinning or mechanical alloying.
    This dataset covers all compositions with an interval of 5 at.% in 59
    binary systems, containing a total of 5483 alloys in the dataset.
    The target property of this dataset is the glass forming ability (GFA),
    i.e. whether the composition can form monolithic glass or not, which is
    either 1 for glass forming or 0 for non-full glass forming.

    References:
        https://pubs.acs.org/doi/10.1021/acs.jpclett.7b01046

    Returns:
        formula (input): chemical formula
        phase (target): only in the "ternary" dataset, designating the phase
                        obtained in glass producing experiments,
                        "AM": amorphous phase
                        "CR": crystalline phase
        gfa (target): glass forming ability, correlated with the phase column,
                      designating whether the composition can form monolithic
                      glass or not,
                      1: glass forming ("AM")
                      0: non-full-glass forming ("CR")

    """
    df = _read_dataframe_from_file(file_path)

    return "glass_binary_v2", df


def _preprocess_expt_formation_enthalpy(file_path):
    """
    Experimental formation enthalpies for inorganic compounds, collected from
    years of calorimetric experiments.
    There are 1,276 entries in this dataset, mostly binary compounds. Matching
    mpids or oqmdids as well as the DFT-computed formation energies are also
    added (if any).

    References:
        https://www.nature.com/articles/sdata2017162

    Returns:
        formula (input): chemical formula
        pearson symbol (input): Pearson symbol of the structure
        space group (input): space group of the structure
        mpid (input): Materials project id (if any)
        oqmdid (input): OQMD id (if any)
        e_form expt (target): experimental formation enthaply (in eV/atom)
        e_form mp (target): formation enthalpy from Materials Project
                            (in eV/atom)
        e_form oqmd (target): formation enthalpy from OQMD (in eV/atom)
    """
    df = _read_dataframe_from_file(file_path)

    return "expt_formation_enthalpy", df


def _preprocess_expt_gap(file_path):
    """
    Experimental band gap of 6354 inorganic semiconductors.

    References:
        https://pubs.acs.org/doi/suppl/10.1021/acs.jpclett.8b00124

    Returns:
        formula (input):
        gap expt (target): band gap (in eV) measured experimentally.
    """
    df = _read_dataframe_from_file(file_path)

    df = df.rename(columns={'composition': 'formula', 'Eg (eV)': 'gap expt'})
    # The numbers in 323 formulas such as 'AgCNO,65' or 'Sr2MgReO6,225' are
    # space group numbers confirmed by Jakoah Brgoch the corresponding author
    df['formula'] = df['formula'].apply(lambda x: x.split(',')[0])

    return "expt_gap", df


def _preprocess_jarvis_dft_2d(file_path):
    """
    Properties of 636 2D materials, most of which (in their bulk forms) are in
    Materials Project. All energy calculations in the refined columns and
    structural relaxations were performed with the optB88-vdw functional.
    Magnetic properties were computed without +U correction.

    References:
        https://www.nature.com/articles/s41598-017-05402-0

    Returns:
        formula (input):
        mpid (input): Corresponding mpid string referring to MP bulk material
        structure (input): Dict representation of pymatgen structure object
        stucture initial (input): Pymatgen structure before relaxation (as dict)
        mu_b (input): Magnetic moment, in terms of bohr magneton

        e_form (target): Formation energy in eV
        gap optb88 (target): Band gap in eV using functional optB88-VDW
        e_exfol (target): Exfoliation energy (monolayer formation E) in eV
    """
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
    """
    Various properties of 25,923 bulk and 2D materials computed with the
    OptB88vdW and TBmBJ functionals taken from the JARVIS DFT database.

    References:
        https://arxiv.org/pdf/1805.07325.pdf
        https://www.nature.com/articles/sdata201882
        https://doi.org/10.1103/PhysRevB.98.014107

    Returns:
        formula (input): chemical formula of compounds
        mpid (input): Materials Project id
        jid (input): JARVIs id
        composition (input):
        structure (input):
        e_exfol (target): exfoliation energy per atom in eV/atom
        e_form (target): formation energy per atom, in eV/atom
        gap opt (target): Band gap calculated with OptB88vDW functional, in eV
        gap tbmbj (target): Band gap calculated with TBMBJ functional, in eV
        mu_b (target): Magnetic moment, in Bohr Magneton
        bulk modulus (target): VRH average calculation of bulk modulus
        shear modulus (target): VRH average calculation of shear modulus
        e mass_x (target): Effective electron mass in x direction (BoltzTraP)
        e mass_y (target): Effective electron mass in y direction (BoltzTraP)
        e mass_z (target): Effective electron mass in z direction (BoltzTraP)
        hole mass_x (target): Effective hole mass in x direction (BoltzTraP)
        hole mass_y (target): Effective hole mass in y direction (BoltzTraP)
        hole mass_z (target): Effective hole mass in z direction (BoltzTraP)
        epsilon_x opt (target): Static dielectric function in x direction
            calculated with OptB88vDW functional.
        epsilon_y opt (target): Static dielectric function in y direction
            calculated with OptB88vDW functional.
        epsilon_z opt (target): Static dielectric function in z direction
            calculated with OptB88vDW functional.
        epsilon_x tbmbj (target): Static dielectric function in x direction
            calculated with TBMBJ functional.
        epsilon_y tbmbj (target): Static dielectric function in y direction
            calculated with TBMBJ functional.
        epsilon_z tbmbj (target): Static dielectric function in z direction
            calculated with TBMBJ functional.
    """
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
    """
    1153 Heusler alloys with DFT-calculated magnetic and electronic properties.
    The 1153 alloys include 576 full, 449 half and 128 inverse Heusler alloys.
    The data are extracted and cleaned (including de-duplicating) from Citrine.

    References:
        https://citrination.com/datasets/150561/

    Returns:
        formula (input): chemical formula
        heusler type (input): Full, Half or Inverse Heusler
        num_electron: No. of electrons per formula unit
        struct type (input): Structure type
        latt const (input): Lattice constant
        tetragonality (input): Tetragonality, i.e. c/a

        e_form (target): Formation energy in eV/atom
        pol fermi (target?): Polarization at Fermi level in %
        mu_b (target): Magnetic moment
        mu_b saturation (target?) Saturation magnetization in emu/cc

        other columns dropped for now:
        gap width: No gap or the gap width value
        stability: True or False, can be inferred from e_form:
                   True if e_form<0, False if e_form>0

    """
    df = _read_dataframe_from_file(file_path)

    dropcols = ['gap width', 'stability']
    df = df.drop(dropcols, axis=1)

    return "heusler_magnetic", df


def _preprocess_steel_strength(file_path):
    """
    312 steels with experimental yield strength and ultimate tensile strength,
    extracted and cleaned (including de-duplicating) from Citrine.

    References:
        https://citrination.com/datasets/153092/

    Returns:
        formula (input): chemical formula
        c (input): weight percent of C
        mn (input): weight percent of Mn
        si (input): weight percent of Si
        cr (input): weight percent of Cr
        ni (input): weight percent of Ni
        mo (input): weight percent of Mo
        v (input): weight percent of V
        n (input): weight percent of N
        nb (input): weight percent of Nb
        co (input): weight percent of Co
        w (input): weight percent of W
        al (input): weight percent of Al
        ti (input): weight percent of Ti
        -These weight percent values of alloying elements are suggested as
         features by a related paper.

        yield strength (target): yield strength in GPa
        tensile strength (target): ultimate tensile strength in GPa
        elongation (target): elongation in %

    """
    df = _read_dataframe_from_file(file_path)

    return "steel_strength", df


def _preprocess_glass_ternary_hipt(file_path):
    """
    Metallic glass formation dataset for ternary alloys, collected from the
    high-throughput sputtering experiments measuring whether it is possible
     to form a glass using sputtering.

    The hipt experimental data are of the Co-Fe-Zr, Co-Ti-Zr, Co-V-Zr and
    Fe-Ti-Nb ternary systems.

    References:
        http://advances.sciencemag.org/content/4/4/eaaq1566

    Returns:
        formula (input): chemical formula
        system (condition): selected system(s)
        processing (condition): "sputtering"
        phase (target): only in the "ternary" dataset, designating the phase
                obtained in glass producing experiments,
                "AM": amorphous phase
                "CR": crystalline phase
        gfa (target): glass forming ability, correlated with the phase column,
                      designating whether the composition can form monolithic
                      glass or not,
                      1: glass forming ("AM")
                      0: non-full-glass forming ("CR")

    """
    df = _read_dataframe_from_file(file_path)

    return "glass_ternary_hipt", df


def _preprocess_jarvis_ml_dft_training(file_path):
    """
    Training data for machine learning algorithms based on the jarvis DFT
    dataset.

    References:
        https://doi.org/10.6084/m9.figshare.6870101.v1

    Returns:
        bulk modulus: VRH average calculation of bulk modulus
        composition: A descriptor of the composition of the material
        e mass_x: Effective electron mass in x direction (BoltzTraP)
        e mass_y: Effective electron mass in y direction (BoltzTraP)
        e mass_z: Effective electron mass in z direction (BoltzTraP)
        e_exfol: exfoliation energy per atom in eV/atom
        e_form: formation energy per atom, in eV/atom
        epsilon_x opt: Static dielectric function in x direction calculated
        with OptB88vDW functional.
        epsilon_x tbmbj: Static dielectric function in x direction calculated
        with TBMBJ functional.
        epsilon_y opt: Static dielectric function in y direction calculated
        with OptB88vDW functional.
        epsilon_y tbmbj: Static dielectric function in y direction calculated
        with TBMBJ functional.
        epsilon_z opt: Static dielectric function in z direction calculated
        with OptB88vDW functional.
        epsilon_z tbmbj: Static dielectric function in z direction calculated
        with TBMBJ functional.
        gap opt: Band gap calculated with OptB88vDW functional, in eV
        gap tbmbj: Band gap calculated with TBMBJ functional, in eV
        hole mass_x: Effective hole mass in x direction (BoltzTraP)
        hole mass_y: Effective hole mass in y direction (BoltzTraP)
        hole mass_z: Effective hole mass in z direction (BoltzTraP)
        jid: JARVIS ID
        mpid: Materials Project ID
        mu_b: Magnetic moment, in Bohr Magneton
        shear modulus: VRH average calculation of shear modulus
        structure: A Pymatgen Structure object describing the crystal structure
        of the material
    """
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
    """
    Loads a pregenerated csv file containing properties of ALL materials in MP
    (approximately 70k). To regenerate the file, use generate_datasets.py.
    To use a version with structures, run generate_mp in generate_datasets.py
    and use the option file_path='mp_all.csv'.

    References:
        https://materialsproject.org/citing

    Returns:
        mpid (input): The Materials Project mpid, as a string.
        formula (input):
        structure (input): The dict of Pymatgen structure object. Only present
            if the csv file containing structure is generated and loaded.
        initial structure (input): The dict of Pymatgen structure object before
            relaxation. Only present if the csv file containing initial
            structure is generated and loaded.

        e_hull (target): The calculated energy above the convex hull, in eV per
            atom
        gap pbe (target): The band gap in eV calculated with PBE-DFT functional
        e_form (target); Formation energy per atom (eV)
        mu_b (target): The total magnetization of the unit cell.
        bulk modulus (target): in GPa, average of Voight, Reuss, and Hill
        shear modulus (target): in GPa, average of Voight, Reuss, and Hill
        elastic anisotropy (target): The ratio of elastic anisotropy.

    Notes:
        If loading the csv with structures, loading will typically take ~10 min
        if using initial structures and about ~3-4 min if only using final
        structures.
    """
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
    """
       Metallic glass formation dataset for ternary alloys, collected from the
       "Nonequilibrium Phase Diagrams of Ternary Amorphous Alloys,’ a volume of
       the Landolt– Börnstein collection.
       This dataset contains experimental measurements of whether it is
       possible to form a glass using a variety of processing techniques at
       thousands of compositions from hundreds of ternary systems.
       The processing techniques are designated in the "processing" column.

       There are originally 7191 experiments in this dataset, will be reduced to
       6203 after deduplicated, and will be further reduced to 6118 if combining
       multiple data for one composition.
       There are originally 6780 melt-spinning experiments in this dataset,
       will be reduced to 5800 if deduplicated, and will be further reduced to
       5736 if combining multiple experimental data for one composition.

       References:
           https://materials.springer.com/bp/docs/978-3-540-47679-5
           https://www.nature.com/articles/npjcompumats201628

       Returns:
           formula (input): chemical formula
           phase (target): only in the "ternary" dataset, designating the phase
                           obtained in glass producing experiments,
                           "AM": amorphous phase
                           "CR": crystalline phase
                           "AC": amorphous-crystalline composite phase
                           "QC": quasi-crystalline phase
           processing (condition): "meltspin" or "sputtering"
           gfa (target): glass forming ability, correlated with the phase
            column, designating whether the composition can form monolithic
            glass or not, 1: glass forming ("AM")
                          0: non-full-glass forming ("CR" or "AC" or "QC")

       """
    df = _read_dataframe_from_file(file_path)

    df.drop_duplicates()
    df["gfa"] = df["phase"].apply(lambda x: 1 if x == "AM" else 0)

    return "glass_ternary_landolt", df


def _preprocess_citrine_thermal_conductivity(file_path):
    """
    Thermal conductivity of 872 compounds measured experimentally and retrieved
    from Citrine database from various references. The reported values are
    measured at various temperatures of which 295 are at room temperature. The
    latter subset is return by default.

    References:
        https://citrineinformatics.github.io/python-citrination-client/

    Returns:
        formula (input): chemical formula of compounds
        k_expt (target): the experimentally measured thermal conductivity in SI
            units of W/m.K
    """
    df = _read_dataframe_from_file(file_path)

    df = df[df['k-units'].isin(
        ['W/m.K', 'W/m$\\cdot$K', 'W/mK', 'W\\m K', 'Wm$^{-1}$K$^{-1}$'])]

    return "citrine_thermal_conductivity", df


def _preprocess_double_perovskites_gap(file_path):
    """
        Band gap of 1306 double perovskites (a_1b_1a_2b_2O6) calculated using ﻿
        Gritsenko, van Leeuwen, van Lenthe and Baerends potential (gllbsc) in
        GPAW.

        References:
            1) https://www.nature.com/articles/srep19375
            2) CMR database: https://cmr.fysik.dtu.dk/

        Returns:
            formula (input): chemical formula w/ sites in the a_1+b_1+a_2+b_2+O6
                order; e.g. in KTaGaTaO6, a_1=="K", b_1=="Ta", a_2=="Ga",
                b_2=="Ta"
            a1/b1/a2/b2 (input): species occupying the corresponding sites.
            gap gllbsc (target): electronic band gap (in eV) calculated via
            gllbsc
        """
    df = pd.read_excel(file_path, sheet_name='bandgap')

    df = df.rename(columns={'A1_atom': 'a_1', 'B1_atom': 'b_1',
                            'A2_atom': 'a_2', 'B2_atom': 'b_2'})
    lumo = pd.read_excel(file_path, sheet_name='lumo')

    return ["double_perovskites_gap", "double_perovskites_gap_lumo"], [df, lumo]


def _preprocess_phonon_dielectric_mp(file_path):
    """
    Phonon (lattice/atoms vibrations) and dielectric properties of 1439
    compounds computed via ABINIT software package in the harmonic
    approximation based on density functional perturbation theory.

    References:
        https://www.nature.com/articles/sdata201865

    Returns:
        mpid (input): The Materials Project mpid, as a string.
        formula (input):
        structure (input):

        eps_total (target): total calculated dielectric constant. Unitless:
            it is a ratio over the dielectric constant at vacuum.
        eps_electronic (target): electronic contribution to the calculated
            dielectric constant; unitless.
        last phdos peak (target): the frequency of the last calculated phonon
            density of states in 1/cm; may be used as an estimation of dominant
            longitudinal optical phonon frequency, a descriptor.

    Notes:
        * Only one of these three targets must be used in a training to prevent
        data leakage.
        * For training, retrieval of formulas and structures via mpids hence
            the usage of composition and structure featurizers is recommended.
    """
    df = _read_dataframe_from_file(file_path)

    df = df[df['asr_breaking'] < 30].drop('asr_breaking', axis=1)
    # remove entries not having structure, formula, or a target
    df = df.dropna()
    df['structure'] = df['structure'].map(ast.literal_eval)

    return 'phonon_dielectric_mp', df.reset_index(drop=True)


def _preprocess_boltztrap_mp(file_path):
    """
        Effective mass and thermoelectric properties of 9036 compounds in The
        Materials Project database that are calculated by the BoltzTraP software
        package run on the GGA-PBE or GGA+U density functional theory
        calculation results. The properties are reported at the temperature of
        300 Kelvin and the carrier concentration of 1e18 1/cm3.

        References:
            https://www.nature.com/articles/sdata201785
            https://contribs.materialsproject.org/carrier_transport/

        Returns:
            mpid (input): The Materials Project mpid, as a string.
            formula (input):
            structure (input):

            m_n (target): n-type/conduction band effective mass. Units: m_e
             where m_e is the mass of an electron; i.e. m_n is a unitless ratio
            m_p (target): p-type/valence band effective mass.
            s_n (target): n-type Seebeck coefficient in micro Volts per Kelvin
            s_p (target): p-type Seebeck coefficient in micro Volts per Kelvin
            pf_n (target): n-type thermoelectric power factor in uW/cm2.K where
                uW is microwatts and a constant relaxation time of 1e-14 assumed
            pf_p (target): p-type power factor in uW/cm2.K

        Note:
            * To avoid data leakage, one may only set the target to one of the
            target columns listed. For example, S_n is strongly correlated with
            PF_n and usually when one is available the other one is available
            too.
            * It is recommended that dos and bandstructure objects are retrieved
            from Materials Porject and then dos, bandstructure and composition
            featurizers are used to generate input features.
        """
    df = _read_dataframe_from_file(file_path, index_col=False)

    df = df.rename(columns={'S_n': 's_n', 'S_p': 's_p',
                            'PF_n': 'pf_n', 'PF_p': 'pf_p'})
    df = df.dropna()
    df['structure'] = df['structure'].map(ast.literal_eval)

    return 'boltztrap_mp', df


def _preprocess_castelli_perovskites(file_path):
    """
        18,928 perovskites generated with ABX combinatorics, calculating gbllsc
        band gap and pbe structure, and also reporting absolute band edge
        positions and heat of formation.

        References:
            http://pubs.rsc.org/en/content/articlehtml/2012/ee/c2ee22341d

        Returns:
            formula (input):
            fermi level (input): in eV
            fermi width (input): fermi bandwidth
            e_form (input): heat of formation (eV)
            gap is direct (input):
            structure (input): crystal structure as dict representing pymatgen
                Structure
            mu_b (input): magnetic moment in terms of Bohr magneton

            gap gllbsc (target): electronic band gap in eV calculated via gllbsc
                functional
            vbm (target): absolute value of valence band edge calculated via
                gllbsc
            cbm (target): similar to vbm but for conduction band
        """
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
    1,181 structures with elastic properties calculated with DFT-PBE.

    References:
        1) https://www.nature.com/articles/sdata20159
        2) https://www.sciencedirect.com/science/article/pii/S0927025618303252

    Returns:
        mpid (input): material id via MP
        formula (input):
        structure (input): dict form of Pymatgen structure
        nsites (input): The number of sites in the structure

        elastic anisotropy (target): ratio of anisotropy of elastic properties
        shear modulus (target): in GPa
        bulk modulus (target): in GPa
        poisson ratio (target):

    Notes:
        This function may return a subset of information which is present in
        load_mp. However, this dataframe is 'clean' with regard to elastic
        properties.
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
    941 structures with piezoelectric properties calculated with DFT-PBE.

    References:
        1) https://www.nature.com/articles/sdata201553
        2) https://www.sciencedirect.com/science/article/pii/S0927025618303252

    Returns:
        mpid (input): material id via MP
        formula (input): string formula
        structure (input): dict form of Pymatgen structure
        nsites (input): The number of sites in the structure

        eij_max (target): Maximum attainable absolute value of the longitudinal
            piezoelectric modulus
        vmax_x/y/z (target): vmax = [vmax_x, vmax_y, vmax_z]. vmax is the
            direction of eij_max (or family of directions, e.g., <111>)
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
    1,056 structures with dielectric properties calculated with DFPT-PBE.

    References:
        1) https://www.nature.com/articles/sdata2016134
        2) https://www.sciencedirect.com/science/article/pii/S0927025618303252

    Returns:
        mpid (input): material id via MP
        formula (input):
        structure (input): dict form of Pymatgen structure
        nsites (input): The number of sites in the structure

        gap pbe (target): Band gap in eV
        refractive index (target): Estimated refractive index
        ep_e poly (target): Polycrystalline electronic contribution to
            dielectric constant (estimate/avg)
        ep poly (target): Polycrystalline dielectric constant (estimate/avg)
        pot. ferroelectric (target): If imaginary optical phonon modes present
            at the Gamma point, the material is potentially ferroelectric
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
    3938 structures and formation energies from "Crystal Structure
    Representations for Machine Learning Models of Formation Energies."

    References:
        1) https://arxiv.org/abs/1503.07406
        2) https://aip.scitation.org/doi/full/10.1063/1.4812323

    Returns:
        mpid (input): material id via MP
        formula (input): string formula
        structure (input): dict form of Pymatgen structure

        e_form (target): Formation energy in eV/atom
        e_hull (target): Energy above hull, in form
    """
    df = _read_dataframe_from_file(file_path, comment="#")

    column_headers = ['material_id', 'e_above_hull', 'formula',
                      'nsites', 'structure', 'formation_energy',
                      'formation_energy_per_atom']
    df['structure'] = pd.Series(
        [Structure.from_dict(ast.literal_eval(s))
         for s in df['structure']], df.index)

    return 'flla', df[column_headers]


# These dictionaries map the filename of datasets to their preprocessors.
# Defaults to just loading in the file with default pd load function for a
# given file type.


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
    "glass_binary_v2": _preprocess_glass_binary_v2,
    "formation_enthalpy_expt": _preprocess_expt_formation_enthalpy,
    "zhuo_gap_expt": _preprocess_expt_gap,
    "jdft_2d-7-7-2018": _preprocess_jarvis_dft_2d,
    "heusler_magnetic": _preprocess_heusler_magnetic,
    "steel_strength": _preprocess_steel_strength,
    "jarvisml_cfid": _preprocess_jarvis_ml_dft_training,
    "glass_ternary_hipt": _preprocess_glass_ternary_hipt,
    "jdft_3d-7-7-2018": _preprocess_jarvis_dft_3d,
    "brgoch_superhard_training": _preprocess_brgoch_superhard_training,
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
