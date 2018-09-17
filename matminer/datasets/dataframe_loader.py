import os
import ast
import hashlib
from collections import namedtuple

import numpy as np
import pandas
import requests

from pymatgen.io.vasp.inputs import Poscar
from pymatgen.core.structure import Structure


__author__ = "Kyle Bystrom <kylebystrom@berkeley.edu>, " \
             "Daniel Dopp <dbdopp@lbl.gov>, " \
             "Anubhav Jain <ajain@lbl.gov>"

# directory to look for data and store datasets can be set by user using
# MATMINER_DATA environment variable, otherwise defaults to module directory
dataset_dir = os.environ.get("MATMINER_DATA",
                             os.path.dirname(os.path.abspath(__file__)))


RemoteFileMetadata = namedtuple("RemoteFileMetadata", ["url", "hash"])


def validate_dataset(data_path, dataset_metadata=None,
                     download_if_missing=True):
    """
    Checks to see if a dataset is on the local machine, if not downloads,
    also checks that the hash of the file data matches that included in the
    metadata

    Args:
        data_path (str): the full path to the file you would like to load,
        if nonexistant will try to download from external source by default

        dataset_metadata (RemoteFileMetadata): a named tuple containing the
        url and hash of the dataset if it is to be downloaded from online

        download_if_missing (bool): whether or not to try downloading the
        dataset if it is not on local disk

    Returns (None)
    """

    # If the file doesn't exist, download it
    if not os.path.exists(data_path):
        if not download_if_missing:
            raise IOError("Data not found and download_if_missing set to False")
        elif dataset_metadata is None:
            raise ValueError("To download an external dataset, the dataset "
                             "metadata must be provided")
        fetch_external_dataset(dataset_metadata.url, data_path)

    # Check to see if downloaded file hash matches the expected value
    sha256hash = hashlib.sha256()
    chunk_size = 8192
    with open(data_path, "rb") as f:
        while True:
            buffer = f.read(chunk_size)
            if not buffer:
                break
            sha256hash.update(buffer)
    file_hash = sha256hash.hexdigest()

    if file_hash != dataset_metadata.hash:
        raise IOError(
            "Error, hash of downloaded file does not match that included in "
            "metadata, the data may be corrupt or altered"
        )


def fetch_external_dataset(url, file_path):
    """
    Downloads file from a given url

    Args:
        url (str): string of where to get external dataset

        file_path (str): string of where to save the file to be retrieved

    Returns (None)
    """

    print("Fetching {} from {} to {}".format(
        os.path.basename(file_path), url, file_path))

    r = requests.get(url, stream=True)

    with open(file_path, "wb") as file_out:
        for chunk in r.iter_content(chunk_size=2048):
            file_out.write(chunk)

    r.close()


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

    data_path = os.path.join(dataset_dir, "elastic_tensor.csv")
    dataset_metadata = RemoteFileMetadata(
        url="https://ndownloader.figshare.com/files/12998804?private_link=d1d110b9ff32460b4f6e",
        hash="f7a18c91fe5dcd51012e5b7e3a37f73aaee9087a036d61bdf9d6464b6fca51a6",
    )

    validate_dataset(data_path, dataset_metadata=dataset_metadata,
                     download_if_missing=download_if_missing)

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

    data_path = os.path.join(dataset_dir, "piezoelectric_tensor.csv")
    dataset_metadata = RemoteFileMetadata(
        url="https://ndownloader.figshare.com/files/12998954?private_link=1266d4fd5e0eafaa7226",
        hash="4be45c8df76a9600f789255ddcb05a92fc3807e0b96fd01e85713a58c34a2ae1"
    )

    validate_dataset(data_path, dataset_metadata=dataset_metadata,
                     download_if_missing=download_if_missing)

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

    data_path = os.path.join(dataset_dir, "dielectric_constant.csv")
    dataset_metadata = RemoteFileMetadata(
        url="https://ndownloader.figshare.com/files/12998735?private_link=a96ce29908cfa82f3f4f",
        hash="ecbd410d33c95d5b05822cff6c7c0ba809a024b4ede3855ec5efc48d5e29ea77",
    )

    validate_dataset(data_path, dataset_metadata=dataset_metadata,
                     download_if_missing=download_if_missing)

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

    data_path = os.path.join(dataset_dir, "flla_2015.csv")
    dataset_metadata = RemoteFileMetadata(
        url="https://ndownloader.figshare.com/files/12998942?private_link=56c924d0d1dda777bef9",
        hash="35b8dbc0b92f4dc7e219fd6606c3a27bee18a9618f376cfee1ff731e306210bb",
    )

    validate_dataset(data_path, dataset_metadata=dataset_metadata,
                     download_if_missing=download_if_missing)

    df = pandas.read_csv(data_path, comment="#")
    column_headers = ['material_id', 'e_above_hull', 'formula',
                      'nsites', 'structure', 'formation_energy',
                      'formation_energy_per_atom']
    df['structure'] = pandas.Series([Structure.from_dict(ast.literal_eval(s))
                                     for s in df['structure']], df.index)
    return df[column_headers]
