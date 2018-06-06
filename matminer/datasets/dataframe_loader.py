import pandas
import os
import ast
import numpy as np
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.core.structure import Structure


__author__ = "Kyle Bystrom <kylebystrom@berkeley.edu>, " \
             "Anubhav Jain <ajain@lbl.gov>"

module_dir = os.path.dirname(os.path.abspath(__file__))


def load_elastic_tensor(include_metadata=False):
    """
    References:
        Jong, M. De, Chen, W., Angsten, T., Jain, A., Notestine, R., Gamst,
        A., Sluiter, M., Ande, C. K., Zwaag, S. Van Der, Plata, J. J., Toher,
        C., Curtarolo, S., Ceder, G., Persson, K. and Asta, M., "Charting the
        complete elastic properties of inorganic crystalline compounds",
        Scientific Data volume 2, Article number: 150009 (2015)

    Args:
        include_metadata (bool): whether to return cif, kpoint_density, poscar

    Returns (pd.DataFrame)
    """
    df = pandas.read_csv(os.path.join(module_dir, "elastic_tensor.csv"),
        comment="#")
    for i in list(df.index):
        for c in ['compliance_tensor', 'elastic_tensor', 'elastic_tensor_original']:
            df.at[(i,c)] = np.array(ast.literal_eval(df.at[(i,c)]))
    df['cif'] = df['structure']
    df['structure'] = pandas.Series([Poscar.from_string(s).structure for s in df['poscar']])
    new_columns = ['material_id', 'formula',
        'nsites', 'space_group', 'volume',
        'structure', 'elastic_anisotropy', 'G_Reuss', 'G_VRH', 'G_Voigt', 'K_Reuss',
        'K_VRH', 'K_Voigt', 'poisson_ratio',
        'compliance_tensor', 'elastic_tensor', 'elastic_tensor_original']
    if include_metadata:
        new_columns += ['cif', 'kpoint_density', 'poscar']
    return df[new_columns]


def load_piezoelectric_tensor(include_metadata=False):
    """
    References:
        de Jong, M., Chen, W., Geerlings, H., Asta, M. & Persson, K. A.
        A database to enable discovery and design of piezoelectric materials.
        Sci. Data 2, 150053 (2015)

    Args:
        include_metadata (bool): whether to return cif, meta, poscar

    Returns (pd.DataFrame)
    """
    df = pandas.read_csv(os.path.join(module_dir, "piezoelectric_tensor.csv"),
        comment="#")
    for i in list(df.index):
        c = 'piezoelectric_tensor'
        df.at[(i,c)] = np.array(ast.literal_eval(df.at[(i,c)]))
    df['cif'] = df['structure']
    df['structure'] = pandas.Series([Poscar.from_string(s).structure for s in df['poscar']])
    new_columns = ['material_id', 'formula',
        'nsites', 'point_group', 'space_group', 'volume',
        'structure', 'eij_max', 'v_max', 'piezoelectric_tensor']
    if include_metadata:
        new_columns += ['cif', 'meta', 'poscar']
    return df[new_columns]


def load_dielectric_constant(include_metadata=False):
    """
    References:
        Petousis, I., Mrdjenovich, D., Ballouz, E., Liu, M., Winston, D.,
        Chen, W., Graf, T., Schladt, T. D., Persson, K. A. & Prinz, F. B.
        High-throughput screening of inorganic compounds for the discovery of
        novel dielectric and optical materials. Sci. Data 4, 160134 (2017).

    Args:
        include_metadata (bool): whether to return cif, meta, poscar

    Returns (pd.DataFrame)
    """
    df = pandas.read_csv(os.path.join(module_dir, "dielectric_constant.csv"),
        comment="#")
    df['cif'] = df['structure']
    df['structure'] = pandas.Series([Poscar.from_string(s).structure for s in df['poscar']])
    new_columns = ['material_id', 'formula',
        'nsites', 'space_group', 'volume',
        'structure',
        'band_gap', 'e_electronic', 'e_total', 'n', 'poly_electronic',
        'poly_total', 'pot_ferroelectric']
    if include_metadata:
        new_columns += ['cif', 'meta', 'poscar']
    return df[new_columns]


def load_flla():
    """
    References:
        1) F. Faber, A. Lindmaa, O.A. von Lilienfeld, R. Armiento,
        "Crystal structure representations for machine learning models of
        formation energies", Int. J. Quantum Chem. 115 (2015) 1094–1101.
        doi:10.1002/qua.24917.

        2) (raw data) Jain, A., Ong, S. P., Hautier, G., Chen, W., Richards, W. D.,
        Dacek, S., Cholia, S., Gunter, D., Skinner, D., Ceder, G. & Persson,
        K. A. Commentary: The Materials Project: A materials genome approach to
        accelerating materials innovation. APL Mater. 1, 11002 (2013).

    Returns (pd.DataFrame)
    """
    df = pandas.read_csv(os.path.join(module_dir, "flla_2015.csv"), comment="#")
    column_headers = ['material_id', 'e_above_hull', 'formula',
                        'nsites', 'structure', 'formation_energy',
                        'formation_energy_per_atom']
    df['structure'] = pandas.Series([Structure.from_dict(ast.literal_eval(s))
        for s in df['structure']], df.index)
    return df[column_headers]
