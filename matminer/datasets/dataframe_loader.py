import os
import ast
import hashlib
from collections import namedtuple

import numpy as np
import pandas
from six.moves.urllib.request import urlretrieve

from pymatgen.io.vasp.inputs import Poscar
from pymatgen.core.structure import Structure


__author__ = "Kyle Bystrom <kylebystrom@berkeley.edu>, " \
             "Daniel Dopp <dbdopp@lbl.gov>, " \
             "Anubhav Jain <ajain@lbl.gov>"

module_dir = os.path.dirname(os.path.abspath(__file__))

RemoteFileMetadata = namedtuple("RemoteFileMetadata", ["url", "hash"])


def fetch_external_dataset(file_metadata, file_path):
    """
    Downloads file from a given url and checks that the hash of the file data
    matches that included in the metadata

    Args:
        file_metadata (RemoteFileMetadata): metadata object which must have url
                                            and hash attributes
        file_path (str): string specifying where to save the file to be
                         retrieved

    Returns (None)
    """
    urlretrieve(file_metadata.url, file_path)

    md5hash = hashlib.md5()
    chunk_size = 8192
    with open(file_path, "rb") as f:
        while True:
            buffer = f.read(chunk_size)
            if not buffer:
                break
            md5hash.update(buffer)
    file_hash = md5hash.hexdigest()

    if file_hash != file_metadata.hash:
        raise IOError(
            "Error, hash of downloaded file does not match that included in "
            "metadata, the data may be corrupt or altered"
        )


def load_elastic_tensor(include_metadata=False, download_if_missing=True):
    """
    References:
        Jong, M. De, Chen, W., Angsten, T., Jain, A., Notestine, R., Gamst,
        A., Sluiter, M., Ande, C. K., Zwaag, S. Van Der, Plata, J. J., Toher,
        C., Curtarolo, S., Ceder, G., Persson, K. and Asta, M., "Charting the
        complete elastic properties of inorganic crystalline compounds",
        Scientific Data volume 2, Article number: 150009 (2015)

    Args:
        include_metadata (bool): whether to return cif, kpoint_density, poscar
        download_if_missing (bool): whether to attempt to download the dataset
                                    from an external source if it is not
                                    on the local machine

    Returns (pd.DataFrame)
    """

    data_path = os.path.join(module_dir, "elastic_tensor.csv")
    dataset_metadata = RemoteFileMetadata(
        url="https://ndownloader.figshare.com/files/12998804",
        hash="f1e16f8cbe01eea97ec891fd361e7add"
    )

    if not os.path.exists(data_path):
        if not download_if_missing:
            raise IOError("Data not found and download_if_missing set to False")

        print("Fetching elastic tensor data from {} to {}".format(
            dataset_metadata.url, data_path))
        fetch_external_dataset(dataset_metadata, data_path)

    df = pandas.read_csv(data_path, comment="#")
    for i in list(df.index):
        for c in ['compliance_tensor', 'elastic_tensor', 'elastic_tensor_original']:
            df.at[(i, c)] = np.array(ast.literal_eval(df.at[(i, c)]))
    df['cif'] = df['structure']
    df['structure'] = pandas.Series([Poscar.from_string(s).structure
                                     for s in df['poscar']])
    new_columns = ['material_id', 'formula', 'nsites', 'space_group', 'volume',
                   'structure', 'elastic_anisotropy', 'G_Reuss', 'G_VRH',
                   'G_Voigt', 'K_Reuss', 'K_VRH', 'K_Voigt', 'poisson_ratio',
                   'compliance_tensor', 'elastic_tensor',
                   'elastic_tensor_original']
    if include_metadata:
        new_columns += ['cif', 'kpoint_density', 'poscar']
    return df[new_columns]


def load_piezoelectric_tensor(include_metadata=False, download_if_missing=True):
    """
    References:
        de Jong, M., Chen, W., Geerlings, H., Asta, M. & Persson, K. A.
        A database to enable discovery and design of piezoelectric materials.
        Sci. Data 2, 150053 (2015)

    Args:
        include_metadata (bool): whether to return cif, meta, poscar
        download_if_missing (bool): whether to attempt to download the dataset
                                    from an external source if it is not
                                    on the local machine

    Returns (pd.DataFrame)
    """
    data_path = os.path.join(module_dir, "piezoelectric_tensor.csv")
    dataset_metadata = RemoteFileMetadata(
        url="https://ndownloader.figshare.com/files/12998954",
        hash="9ca3a5e9f91dbb0302d0a60fb6d675d7"
    )

    if not os.path.exists(data_path):
        if not download_if_missing:
            raise IOError("Data not found and download_if_missing set to False")

        print("Fetching piezoelectric tensor data from {} to {}".format(
            dataset_metadata.url, data_path))
        fetch_external_dataset(dataset_metadata, data_path)

    df = pandas.read_csv(data_path, comment="#")
    for i in list(df.index):
        c = 'piezoelectric_tensor'
        df.at[(i, c)] = np.array(ast.literal_eval(df.at[(i, c)]))
    df['cif'] = df['structure']
    df['structure'] = pandas.Series([Poscar.from_string(s).structure
                                     for s in df['poscar']])
    new_columns = ['material_id', 'formula', 'nsites', 'point_group',
                   'space_group', 'volume', 'structure', 'eij_max', 'v_max',
                   'piezoelectric_tensor']
    if include_metadata:
        new_columns += ['cif', 'meta', 'poscar']
    return df[new_columns]


def load_dielectric_constant(include_metadata=False, download_if_missing=True):
    """
    References:
        Petousis, I., Mrdjenovich, D., Ballouz, E., Liu, M., Winston, D.,
        Chen, W., Graf, T., Schladt, T. D., Persson, K. A. & Prinz, F. B.
        High-throughput screening of inorganic compounds for the discovery of
        novel dielectric and optical materials. Sci. Data 4, 160134 (2017).

    Args:
        include_metadata (bool): whether to return cif, meta, poscar
        download_if_missing (bool): whether to attempt to download the dataset
                                    from an external source if it is not
                                    on the local machine

    Returns (pd.DataFrame)
    """

    data_path = os.path.join(module_dir, "dielectric_constant.csv")
    dataset_metadata = RemoteFileMetadata(
        url="https://ndownloader.figshare.com/files/12998735",
        hash="9ab2afa0e17ecbe6b8862c7cae88634f"
    )

    if not os.path.exists(data_path):
        if not download_if_missing:
            raise IOError("Data not found and download_if_missing set to False")

        print("Fetching dielectric constant data from {} to {}".format(
            dataset_metadata.url, data_path))
        fetch_external_dataset(dataset_metadata, data_path)

    df = pandas.read_csv(data_path, comment="#")
    df['cif'] = df['structure']
    df['structure'] = pandas.Series([Poscar.from_string(s).structure
                                     for s in df['poscar']])
    new_columns = ['material_id', 'formula', 'nsites', 'space_group', 'volume',
                   'structure', 'band_gap', 'e_electronic', 'e_total', 'n',
                   'poly_electronic', 'poly_total', 'pot_ferroelectric']
    if include_metadata:
        new_columns += ['cif', 'meta', 'poscar']
    return df[new_columns]


def load_flla(download_if_missing=True):
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

     Args:
        download_if_missing (bool): whether to attempt to download the dataset
                                    from an external source if it is not
                                    on the local machine

    Returns (pd.DataFrame)
    """
    data_path = os.path.join(module_dir, "flla_2015.csv")
    dataset_metadata = RemoteFileMetadata(
        url="https://ndownloader.figshare.com/files/12998942",
        hash="96430434ab143fad685eece5a2340c6f"
    )

    if not os.path.exists(data_path):
        if not download_if_missing:
            raise IOError("Data not found and download_if_missing set to False")

        print("Fetching flla data from {} to {}".format(
            dataset_metadata.url, data_path))
        fetch_external_dataset(dataset_metadata, data_path)

    df = pandas.read_csv(data_path, comment="#")
    column_headers = ['material_id', 'e_above_hull', 'formula',
                      'nsites', 'structure', 'formation_energy',
                      'formation_energy_per_atom']
    df['structure'] = pandas.Series([Structure.from_dict(ast.literal_eval(s))
                                     for s in df['structure']], df.index)
    return df[column_headers]
