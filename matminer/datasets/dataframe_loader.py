import pandas
import os
import ast
import numpy as np
from pymatgen.io.vasp.inputs import Poscar


__author__ = "Kyle Bystrom <kylebystrom@berkeley.edu>, " \
             "Anubhav Jain <ajain@lbl.gov>"

module_dir = os.path.dirname(os.path.abspath(__file__))


def load_elastic_tensor(include_metadata = False):
    # ref: Jong, M. De, Chen, W., Angsten, T., Jain, A., Notestine, R., Gamst,
    # A., Sluiter, M., Ande, C. K., Zwaag, S. Van Der, Plata, J. J., Toher,
    # C., Curtarolo, S., Ceder, G., Persson, K. a & Asta, M. Charting the
    # complete elastic properties of inorganic crystalline compounds. Sci.
    # Data 2, 150009 (2015).
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

def load_piezoelectric_tensor(include_metadata = False):
    # ref: de Jong, M., Chen, W., Geerlings, H., Asta, M. & Persson, K. A.
    # A database to enable discovery and design of piezoelectric materials.
    # Sci. Data 2, 150053 (2015).
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

def load_dielectric_constant(include_metadata = False):
    # ref: Petousis, I., Mrdjenovich, D., Ballouz, E., Liu, M., Winston, D.,
    # Chen, W., Graf, T., Schladt, T. D., Persson, K. A. & Prinz, F. B.
    # High-throughput screening of inorganic compounds for the discovery of
    # novel dielectric and optical materials. Sci. Data 4, 160134 (2017).
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
