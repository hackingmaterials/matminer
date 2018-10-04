import os
import ast
import hashlib
import warnings

import numpy as np
import pandas
import requests
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.core.structure import Structure


__author__ = "Kyle Bystrom <kylebystrom@berkeley.edu>, " \
             "Anubhav Jain <ajain@lbl.gov>" \
             "Daniel Dopp <dbdopp@lbl.gov>"

_dataset_dict = {
    'flla': {
        'file_name':
            'flla_2015.csv',
        'url':
            'https://ndownloader.figshare.com/files/13082810',
        'hash':
            '35b8dbc0b92f4dc7e219fd6606c3a27bee18a9618f376cfee1ff731e306210bb',
        'reference':
            """
            1) F. Faber, A. Lindmaa, O.A. von Lilienfeld, R. Armiento,
            "Crystal structure representations for machine learning models of
            formation energies", Int. J. Quantum Chem. 115 (2015) 1094–1101.
            doi:10.1002/qua.24917.

            (raw data)
            2) Jain, A., Ong, S. P., Hautier, G., Chen, W., Richards, W. D.,
            Dacek, S., Cholia, S., Gunter, D., Skinner, D., Ceder, G. & Persson,
            K. A. Commentary: The Materials Project: A materials genome approach 
            to accelerating materials innovation. APL Mater. 1, 11002 (2013).
            """,
        'description':
            "3938 structures and computed formation energies from "
            "\"Crystal Structure Representations for Machine Learning Models "
            "of Formation Energies.\"",
        'columns': {
            'material_id': "Materials Project ID of the material",

            'formula': "Chemical formula of the material",

            'e_above_hull': "The energy of decomposition of this material "
                            "into the set of most stable materials at this "
                            "chemical composition, in eV/atom.",

            'nsites': "The \# of atoms in the unit cell of the calculation.",

            'structure': "pandas Series defining the structure of the material",

            'formation_energy_per_atom': "See formation_energy",

            'formation_energy': "Computed formation energy at 0K, "
                                "0atm using a reference state of zero "
                                "for the pure elements.",
        },
        'bibtex_refs': [
            """@article{doi:10.1002/qua.24917,
            author = {Faber, Felix and Lindmaa, Alexander and von Lilienfeld, 
            O. Anatole and Armiento, Rickard},
            title = {Crystal structure representations for machine learning 
            models of formation energies},
            journal = {International Journal of Quantum Chemistry},
            volume = {115},
            number = {16},
            pages = {1094-1101},
            keywords = {machine learning, formation energies, representations, 
            crystal structure, periodic systems},
            doi = {10.1002/qua.24917},
            url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/qua.24917},
            eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1002/qua.24917},
            abstract = {We introduce and evaluate a set of feature vector 
            representations of crystal structures for machine learning (ML) 
            models of formation energies of solids. ML models of atomization 
            energies of organic molecules have been successful using a Coulomb
            matrix representation of the molecule. We consider three ways to 
            generalize such representations to periodic systems: (i) a matrix 
            where each element is related to the Ewald sum of the electrostatic 
            interaction between two different atoms in the unit cell repeated 
            over the lattice; (ii) an extended Coulomb-like matrix that takes 
            into account a number of neighboring unit cells; and (iii) an 
            ansatz that mimics the periodicity and the basic features of the 
            elements in the Ewald sum matrix using a sine function of the 
            crystal coordinates of the atoms. The representations are compared 
            for a Laplacian kernel with Manhattan norm, trained to reproduce 
            formation energies using a dataset of 3938 crystal structures 
            obtained from the Materials Project. For training sets consisting 
            of 3000 crystals, the generalization error in predicting formation 
            energies of new structures corresponds to (i) 0.49, (ii) 0.64, and 
            (iii) for the respective representations. © 2015 Wiley Periodicals, 
            Inc.}
            }
            """,

            """
            @article{doi:10.1063/1.4812323,
            author = {Jain,Anubhav  and Ong,Shyue Ping  and Hautier,Geoffroy  
            and Chen,Wei  and Richards,William Davidson  and Dacek,Stephen  
            and Cholia,Shreyas  and Gunter,Dan  and Skinner,David  
            and Ceder,Gerbrand  and Persson,Kristin A. },
            title = {Commentary: The Materials Project: A materials genome 
            approach to accelerating materials innovation},
            journal = {APL Materials},
            volume = {1},
            number = {1},
            pages = {011002},
            year = {2013},
            doi = {10.1063/1.4812323},
            URL = {https://doi.org/10.1063/1.4812323},
            eprint = {https://doi.org/10.1063/1.4812323}
            }
            """,
        ],
        'num_entries': 3938,
    },

    'elastic_tensor_2015': {
        'file_name':
            'elastic_tensor.csv',
        'url':
            'https://ndownloader.figshare.com/files/13082813',
        'hash':
            'f7a18c91fe5dcd51012e5b7e3a37f73aaee9087a036d61bdf9d6464b6fca51a6',
        'reference':
            """
            Jong, M. De, Chen, W., Angsten, T., Jain, A., Notestine, R., Gamst,
            A., Sluiter, M., Ande, C. K., Zwaag, S. Van Der, Plata, J. J., Toher,
            C., Curtarolo, S., Ceder, G., Persson, K. and Asta, M., "Charting 
            the complete elastic properties of inorganic crystalline compounds",
            Scientific Data volume 2, Article number: 150009 (2015)
            """,
        'description':
            "1,180 structures with elastic properties calculated with DFT-PBE.",
        'columns': {
            'material_id': "Materials Project ID of the material",

            'formula': "Chemical formula of the material",

            'nsites': "The \# of atoms in the unit cell of the calculation.",

            'space_group': "Integer specifying the crystallographic structure "
                           "of the material",

            'volume': "Volume of the unit cell in cubic angstroms, "
                      "For supercell calculations, this quantity refers "
                      "to the volume of the full supercell. ",

            'structure': "pandas Series defining the structure of the material",

            'elastic_anisotropy': "measure of directional dependence of the "
                                  "materials elasticity, metric is always >= 0",

            'G_Reuss': "Lower bound on shear modulus for "
                       "polycrystalline material",

            'G_VRH': "Average of G_Reuss and G_Voigt",

            'G_Voigt': "Upper bound on shear modulus for "
                       "polycrystalline material",

            'K_Reuss': "Lower bound on bulk modulus for "
                       "polycrystalline material",

            'K_VRH': "Average of K_Reuss and K_Voigt",

            'K_Voigt': "Upper bound on bulk modulus for "
                       "polycrystalline material",

            'poisson_ratio': "Describes lateral response to loading",

            'compliance_tensor': "Tensor describing elastic behavior",

            'elastic_tensor': "Tensor describing elastic behavior "
                              "corresponding to IEEE orientation, "
                              "symmetrized to crystal structure ",

            'elastic_tensor_original': "Tensor describing elastic behavior, "
                                       "unsymmetrized, corresponding to POSCAR "
                                       "conventional standard cell orientation",

            'cif': "optional: Description string for structure",

            'kpoint_density': "optional: Sampling parameter from calculation",

            'poscar': "optional: Poscar metadata",
        },
        'bibtex_refs': [
            """
            @Article{deJong2015,
            author={de Jong, Maarten and Chen, Wei and Angsten, Thomas 
            and Jain, Anubhav and Notestine, Randy and Gamst, Anthony
            and Sluiter, Marcel and Krishna Ande, Chaitanya 
            and van der Zwaag, Sybrand and Plata, Jose J. and Toher, Cormac
            and Curtarolo, Stefano and Ceder, Gerbrand and Persson, Kristin A.
            and Asta, Mark},
            title={Charting the complete elastic properties 
            of inorganic crystalline compounds},
            journal={Scientific Data},
            year={2015},
            month={Mar},
            day={17},
            publisher={The Author(s)},
            volume={2},
            pages={150009},
            note={Data Descriptor},
            url={http://dx.doi.org/10.1038/sdata.2015.9}
            }
            """,
        ],
        'num_entries': 1180,
    },

    'piezoelectric_tensor': {
        'file_name':
            'piezoelectric_tensor.csv',
        'url':
            'https://ndownloader.figshare.com/files/13082804',
        'hash':
            '4be45c8df76a9600f789255ddcb05a92fc3807e0b96fd01e85713a58c34a2ae1',
        'reference':
            """
            de Jong, M., Chen, W., Geerlings, H., Asta, M. & Persson, K. A.
            A database to enable discovery and design of piezoelectric materials.
            Sci. Data 2, 150053 (2015)
            """,
        'description':
            "941 structures with piezoelectric properties,"
            " calculated with DFT-PBE.",
        "columns": {
            'material_id': "Materials Project ID of the material",

            'formula': "Chemical formula of the material",

            'nsites': "The \# of atoms in the unit cell of the calculation.",

            'point_group': "Descriptor of crystallographic structure of the "
                           "material",

            'space_group': "Integer specifying the crystallographic structure "
                           "of the material",

            'volume': "Volume of the unit cell in cubic angstroms, "
                      "For supercell calculations, this quantity refers "
                      "to the volume of the full supercell. ",

            'structure': "pandas Series defining the structure of the material",

            'eij_max': "Piezoelectric modulus",

            'v_max': "Crystallographic direction",

            'piezoelectric_tensor': "Tensor describing the piezoelectric"
                                    " properties of the material",

            'cif': "optional: Description string for structure",

            'meta': "optional, metadata descriptor of the datapoint",

            'poscar': "optional: Poscar metadata",
        },
        'bibtex_refs': [
            """
            @Article{deJong2015,
            author={de Jong, Maarten and Chen, Wei and Geerlings, Henry
            and Asta, Mark and Persson, Kristin Aslaug},
            title={A database to enable discovery and design of piezoelectric 
            materials},
            journal={Scientific Data},
            year={2015},
            month={Sep},
            day={29},
            publisher={The Author(s)},
            volume={2},
            pages={150053},
            note={Data Descriptor},
            url={http://dx.doi.org/10.1038/sdata.2015.53}
            }
            """,
        ],
        'num_entries': 941
    },

    'dielectric_constant': {
        'file_name':
            'dielectric_constant.csv',
        'url':
            'https://ndownloader.figshare.com/files/13082816',
        'hash':
            'ecbd410d33c95d5b05822cff6c7c0ba809a024b4ede3855ec5efc48d5e29ea77',
        'reference':
            """
            Petousis, I., Mrdjenovich, D., Ballouz, E., Liu, M., Winston, D.,
            Chen, W., Graf, T., Schladt, T. D., Persson, K. A. & Prinz, F. B.
            High-throughput screening of inorganic compounds for the discovery 
            of novel dielectric and optical materials. Sci. Data 4, 160134 (2017).
            """,
        'description':
            "1,056 structures with dielectric properties,"
            " calculated with DFPT-PBE.",
        'columns': {
            'material_id': "Materials Project ID of the material",

            'formula': "Chemical formula of the material",

            'nsites': "The \# of atoms in the unit cell of the calculation.",

            'space_group': "Integer specifying the crystallographic structure "
                           "of the material",

            'volume': "Volume of the unit cell in cubic angstroms, "
                      "For supercell calculations, this quantity refers "
                      "to the volume of the full supercell. ",

            'structure': "pandas Series defining the structure of the material",

            'band_gap': "Measure of the conductivity of a material",

            'e_electronic': "electronic contribution to dielectric tensor",

            'e_total': "Total dielectric tensor incorporating "
                       "both electronic and ionic contributions",

            'n': "Refractive Index",

            'poly_electronic': "the average of the eigenvalues of the "
                               "electronic contribution to the "
                               "dielectric tensor",

            'poly_total': "the average of the eigenvalues of the total "
                          "(electronic and ionic) contributions to the "
                          "dielectric tensor",

            'pot_ferroelectric': "Whether the material is "
                                 "potentially ferroelectric",

            'cif': "optional: Description string for structure",

            'meta': "optional, metadata descriptor of the datapoint",

            'poscar': "optional: Poscar metadata",
        },
        'bibtex_refs': [
            """
            @Article{Petousis2017,
            author={Petousis, Ioannis and Mrdjenovich, David and Ballouz, Eric
            and Liu, Miao and Winston, Donald and Chen, Wei and Graf, Tanja
            and Schladt, Thomas D. and Persson, Kristin A. and Prinz, Fritz B.},
            title={High-throughput screening of inorganic compounds for the 
            discovery of novel dielectric and optical materials},
            journal={Scientific Data},
            year={2017},
            month={Jan},
            day={31},
            publisher={The Author(s)},
            volume={4},
            pages={160134},
            note={Data Descriptor},
            url={http://dx.doi.org/10.1038/sdata.2016.134}
            }
            """,
        ],
        'num_entries': 1056,
    },
}


def load_dataset(name, data_home=None, download_if_missing=True,
                 include_metadata=False, **kws):
    """
    Loads a dataframe containing the dataset specified with the 'name' field.

    Dataset file is stored/loaded from data_home if specified, otherwise at
    the MATMINER_DATA environment variable if set or at matminer/datasets
    by default.

    Args:
        name (str): keyword specifying what dataset to load, use
            available_datasets() for options

        data_home (str): path to folder to look for dataset file

        download_if_missing (bool): whether to download the dataset if is not
            found on disk

        include_metadata (bool): optional argument for some datasets with
            metadata fields

        **kws: additional arguments to be passed to the dataset file reader

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
                             dataset_metadata['file_name'])
    _validate_dataset(data_path, dataset_metadata['url'],
                      dataset_metadata['hash'], download_if_missing)

    df = pandas.read_csv(data_path, comment="#", **kws)

    if name == "elastic_tensor":
        for i in list(df.index):
            for c in ['compliance_tensor', 'elastic_tensor',
                      'elastic_tensor_original']:
                df.at[(i, c)] = np.array(ast.literal_eval(df.at[(i, c)]))
        df['cif'] = df['structure']
        df['structure'] = pandas.Series([Poscar.from_string(s).structure
                                         for s in df['poscar']])
        new_columns = ['material_id', 'formula', 'nsites', 'space_group',
                       'volume',
                       'structure', 'elastic_anisotropy', 'G_Reuss', 'G_VRH',
                       'G_Voigt', 'K_Reuss', 'K_VRH', 'K_Voigt',
                       'poisson_ratio',
                       'compliance_tensor', 'elastic_tensor',
                       'elastic_tensor_original']
        if include_metadata:
            new_columns += ['cif', 'kpoint_density', 'poscar']
        return df[new_columns]

    elif name == "piezoelectric_tensor":
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

    elif name == "dielectric_constant":
        df['cif'] = df['structure']
        df['structure'] = pandas.Series([Poscar.from_string(s).structure
                                         for s in df['poscar']])
        new_columns = ['material_id', 'formula', 'nsites', 'space_group',
                       'volume',
                       'structure', 'band_gap', 'e_electronic', 'e_total', 'n',
                       'poly_electronic', 'poly_total', 'pot_ferroelectric']
        if include_metadata:
            new_columns += ['cif', 'meta', 'poscar']
        return df[new_columns]

    elif name == "flla":
        df = pandas.read_csv(data_path, comment="#")
        column_headers = ['material_id', 'e_above_hull', 'formula',
                          'nsites', 'structure', 'formation_energy',
                          'formation_energy_per_atom']
        df['structure'] = pandas.Series(
            [Structure.from_dict(ast.literal_eval(s))
             for s in df['structure']], df.index)
        return df[column_headers]

    else:
        return df


def available_datasets(print_datasets=True,
                       print_descriptions=True,
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

    if sort_method not in {"alphabetic", "num_entries"}:
        raise ValueError("Error, unsupported sorting metric,"
                         " see docs for options")

    if sort_method == 'num_entries':
        dataset_names = sorted(dataset_dict.keys(),
                               key=lambda x: dataset_dict[x]["num_entries"],
                               reverse=True)
    else:
        dataset_names = sorted(dataset_dict.keys())

    # If checks done before for loop to avoid unnecessary repetitive evaluation
    if print_datasets and print_descriptions:
        for name in dataset_names:
            print(name, dataset_dict[name]["description"], "", sep="\n")
    elif print_datasets:
        for name in dataset_names:
            print(name)

    return list(dataset_names)


def _load_dataset_dict():
    """
    Loads the dataset dictionary, currently just returns dict,
    will in the future load a file.

    Returns: (dict)
    """
    return _dataset_dict


def _get_data_home(data_home=None):
    """
    Selects the home directory to look for datasets, if the specified home
    directory doesn't exist the directory structure is built
    Args:
        data_home (str): folder to look in, if None a default is selected

    Returns (str)
    """

    # If user doesn't specify a dataset directory: first check for env var,
    # then default to the "matminer/datasets/" package folder
    if data_home is None:
        data_home = os.environ.get("MATMINER_DATA",
                                   os.path.dirname(os.path.abspath(__file__)))

    data_home = os.path.expanduser(data_home)

    return data_home


def _validate_dataset(data_path, url=None, file_hash=None,
                      download_if_missing=True):
    """
    Checks to see if a dataset is on the local machine,
    if not tries to download if download_if_missing is set to true,
    also checks that the hash of the file data matches that included in the
    metadata

    Args:
        data_path (str): the full path to the file you would like to load,
        if nonexistent will try to download from external source by default

        url (str): a string specifying the url to fetch the dataset from if it
        is not available

        file_hash (str): hash of file used to check for file integrity

        download_if_missing (bool): whether or not to try downloading the
        dataset if it is not on local disk

    Returns (None)
    """

    # If the file doesn't exist, download it
    if not os.path.exists(data_path):

        # Ensure proper arguments for download
        if not download_if_missing:
            raise IOError("Data not found and download_if_missing set to False")
        elif url is None:
            raise ValueError("To download an external dataset, the url "
                             "metadata must be provided")

        # Ensure storage location exists
        data_home = os.path.dirname(data_path)

        if not os.path.exists(data_home):
            print("Making dataset storage folder at {}".format(data_home))
            os.makedirs(data_home)

        _fetch_external_dataset(url, data_path)

    # Check to see if file hash matches the expected value, if hash is provided
    if file_hash is not None:
        sha256hash = hashlib.sha256()
        chunk_size = 8192
        with open(data_path, "rb") as f:
            while True:
                buffer = f.read(chunk_size)
                if not buffer:
                    break
                sha256hash.update(buffer)
        computed_hash = sha256hash.hexdigest()

        if file_hash != computed_hash:
            raise UserWarning(
                "Error, hash of downloaded file does not match that "
                "included in metadata, the data may be corrupt or altered"
            )


def _fetch_external_dataset(url, file_path):
    """
    Downloads file from a given url

    Args:
        url (str): string of where to get external dataset

        file_path (str): string of where to save the file to be retrieved

    Returns (None)
    """

    # Fetch data from given url
    print("Fetching {} from {} to {}".format(
        os.path.basename(file_path), url, file_path))

    r = requests.get(url, stream=True)

    with open(file_path, "wb") as file_out:
        for chunk in r.iter_content(chunk_size=2048):
            file_out.write(chunk)

    r.close()


# Turns on Deprecation Warning printing for the four load_x functions below
warnings.filterwarnings('always',
                        message="This function has been deprecated, .*",
                        category=DeprecationWarning)


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
    warnings.warn("This function has been deprecated, "
                  "please use load_dataset('elastic_tensor_2015') instead",
                  DeprecationWarning)

    return load_dataset('elastic_tensor_2015',
                        include_metadata=include_metadata,
                        download_if_missing=download_if_missing)


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

    warnings.warn("This function has been deprecated, "
                  "please use load_dataset('piezoelectric_tensor') instead",
                  DeprecationWarning)

    return load_dataset('piezoelectric_tensor',
                        include_metadata=include_metadata,
                        download_if_missing=download_if_missing)


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

    warnings.warn("This function has been deprecated, "
                  "please use load_dataset('dielectric_constant') instead",
                  DeprecationWarning)

    return load_dataset('dielectric_constant',
                        include_metadata=include_metadata,
                        download_if_missing=download_if_missing)


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

    warnings.warn("This function has been deprecated, "
                  "please use load_dataset('flla') instead",
                  DeprecationWarning)

    return load_dataset('flla', download_if_missing=download_if_missing)
