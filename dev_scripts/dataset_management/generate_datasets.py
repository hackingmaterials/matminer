from time import sleep
from math import ceil

import pandas as pd
from tqdm import tqdm
from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition

from matminer.utils.io import store_dataframe_as_json
from matminer.featurizers.conversions import DictToObject, \
    StructureToOxidStructure, StrToComposition, CompositionToOxidComposition
from matminer.data_retrieval.retrieve_MP import MPDataRetrieval, MPRestError


def convert_to_oxide_structure(df, structure_col_name="structure",
                               batch_size=1000):
    """
    Takes a dataframe with a pymatgen Structure column and adds oxidation data
    to it. If structure is not a proper object, will try to convert first.

    Args:
        df (pd.DataFrame): DataFrame with a column of Structure objects

        structure_col_name (str): The identifier of the structure column

        batch_size (int): Size of batches to process dataframe in

    Returns: (pd.DataFrame)
    """
    if batch_size is None:
        batch_size = len(df)

    if not isinstance(df[structure_col_name][0], Structure):
        df = DictToObject(
            target_col_id=structure_col_name, overwrite_data=True
        ).featurize_dataframe(df, structure_col_name)

    oxide_featurizer = StructureToOxidStructure(
            target_col_id=structure_col_name, overwrite_data=True
        )

    # Process in batches to give usable progress bar
    for i in tqdm(range(ceil(len(df) / batch_size)),
                  desc="Processing oxidation state of structures in batches"):
        start_index = i * batch_size
        end_index = start_index + batch_size

        df.iloc[start_index:end_index] = oxide_featurizer.featurize_dataframe(
            df.iloc[start_index:end_index], structure_col_name, pbar=False
        )

    return df


def convert_to_oxide_composition(df, composition_col_name="composition",
                                 batch_size=1000):
    """
    Takes a dataframe with a pymatgen Composition column and adds oxidation data
    to it. If composition is not a proper object, will try to convert first.

    Args:
        df (pd.DataFrame): DataFrame with a column of Composition objects

        composition_col_name (str): The identifier of the composition column

        batch_size (int): Size of batches to process dataframe in

    Returns: (pd.DataFrame)
    """
    if batch_size is None:
        batch_size = len(df)

    if not isinstance(df[composition_col_name][0], Composition):
        df = StrToComposition(
            target_col_id=composition_col_name, overwrite_data=True
        ).featurize_dataframe(df, composition_col_name)

    oxide_featurizer = CompositionToOxidComposition(
        target_col_id=composition_col_name, overwrite_data=True
    )

    # Process in batches to give usable progress bar
    for i in tqdm(range(ceil(len(df) / batch_size)),
                  desc="Processing oxidation state of compositions in batches"):
        start_index = i * batch_size
        end_index = start_index + batch_size

        df.iloc[start_index:end_index] = oxide_featurizer.featurize_dataframe(
            df.iloc[start_index:end_index], composition_col_name, pbar=False
        )

    return df


# Functions for generating static datasets from various database resources
def generate_mp(max_nsites=None, properties=None, write_to_csv=False,
                write_to_compressed_json=False, include_oxide_info=True):
    """
    Grabs all mp materials. This will create two csv/json.gz files:
        * mp_nostruct: All MP materials, not including structures
        * mp_all: All MP materials, including structures

    Args:
        max_nsites (int): The maximum number of sites to include in the query.

        properties (iterable of strings): list of properties supported by
            MPDataRetrieval

        write_to_csv (bool): whether to write resulting dataframe to csv

        write_to_compressed_json (bool): whether to write resulting
            dataframe to json.gz file

        include_oxide_info (bool): Whether to convert pymatgen objects to
            objects including oxidation state information

    Returns (pandas.DataFrame):
        retrieved/generated data without structure data
    """

    # Set default properties if None and ensure is a list
    if properties is None:
        properties = ['pretty_formula', 'e_above_hull', 'band_gap',
                      'total_magnetization', 'elasticity.elastic_anisotropy',
                      'elasticity.K_VRH', 'elasticity.G_VRH', 'structure',
                      'energy', 'energy_per_atom', 'formation_energy_per_atom']
    elif not isinstance(properties, list):
        properties = list(properties)

    # Pick columns to drop structure data from
    drop_cols = []
    for col_name in ["structure", "initial_structure"]:
        if col_name in properties:
            drop_cols.append(col_name)

    mpdr = MPDataRetrieval()
    if max_nsites is not None:
        sites_list = [i for i in range(1, max_nsites + 1)]
    else:
        sites_list = [i for i in range(1, 101)] + [{"$gt": 100}]

    df = pd.DataFrame()
    for site_specifier in tqdm(sites_list, desc="Querying Materials Project"):
        # While loop to repeat queries if server request fails
        while True:
            try:
                site_response = mpdr.get_dataframe(
                    criteria={"nsites": site_specifier},
                    properties=properties, index_mpid=True
                )
                break

            except MPRestError:
                tqdm.write("Error querying materials project, "
                           "trying again after 5 sec")
                sleep(5)

        df = df.append(site_response)

    df.rename(columns={'elasticity.K_VRH': 'K_VRH',
                       'elasticity.G_VRH': 'G_VRH',
                       'pretty_formula': 'composition'},
              index=str, inplace=True)

    # Convert returned data to the appropriate
    # pymatgen Structure/Composition objects
    df = DictToObject(
        target_col_id="structure", overwrite_data=True
    ).featurize_dataframe(df, "structure")

    df = StrToComposition(
        target_col_id="composition", overwrite_data=True
    ).featurize_dataframe(df, "composition")

    if include_oxide_info:
        df = convert_to_oxide_structure(df)
        df = convert_to_oxide_composition(df)

    tqdm.write("DataFrame with {} entries created".format(len(df)))

    # Write data out to file if user so chooses
    if write_to_csv:
        df.to_csv("mp_all.csv")
        df.drop(drop_cols, axis=1, inplace=True)
        df.to_csv("mp_nostruct.csv")

    if write_to_compressed_json:
        store_dataframe_as_json(df, "mp_all.json.gz", compression="gz")
        df = df.drop(drop_cols, axis=1)
        store_dataframe_as_json(df, "mp_nostruct.json.gz", compression="gz")

    return df


def generate_elastic_tensor(write_to_csv=False, write_to_compressed_json=False,
                            include_oxide_info=True):
    """
    Grabs all materials with elasticity data.
    This will return a csv/json.gz file:
        * elastic_tensor_2018: All MP materials with elasticity data,
            including structures

    Args:
        write_to_csv (bool): whether to write resulting dataframe to csv

        write_to_compressed_json (bool): whether to write resulting
            dataframe to json.gz file

        include_oxide_info (bool): Whether to add oxidation info to pymatgen
            objects

    Returns (pandas.DataFrame):
        retrieved/generated data
    """

    # ignore mp-978085 as causes Python to
    # crash when performing Voronoi analysis.
    criteria = {
        "elasticity": {"$exists": True},
        'material_id': {'$ne': 'mp-978085'}
    }
    properties = ['structure', 'pretty_formula',
                  'elasticity.K_VRH', 'elasticity.G_VRH',
                  'elasticity.warnings'
                  ]

    df = pd.DataFrame()
    mpdr = MPDataRetrieval()

    # Iterate over each site number to ensure returned object isn't too large
    for site_num in tqdm([i for i in range(1, 101)] + [{"$gt": 100}],
                         desc="Querying MP"):
        criteria["nsites"] = site_num
        # While loop to repeat queries if server request fails
        while True:
            try:
                site_response = mpdr.get_dataframe(
                    criteria=criteria, properties=properties, index_mpid=True
                )
                break

            except MPRestError:
                tqdm.write("Error querying materials project, "
                           "trying again after 5 sec")
                sleep(5)

        df = df.append(site_response)

    df.rename(columns={'elasticity.K_VRH': 'K_VRH',
                       'elasticity.G_VRH': 'G_VRH',
                       'pretty_formula': 'composition'},
              index=str, inplace=True)

    tqdm.write("There are {} elastic entries on MP".format(
        df['K_VRH'].count()
    ))

    df = df[~(df["elasticity.warnings"].apply(bool))]
    df = df.drop(["elasticity.warnings"], axis=1)

    tqdm.write("There are {} elastic entries on MP with no warnings".format(
        df['K_VRH'].count()
    ))

    print(df.columns)

    # Convert returned data to the appropriate
    # pymatgen Structure/Composition objects
    df = DictToObject(
        target_col_id="structure", overwrite_data=True
    ).featurize_dataframe(df, "structure")

    df = StrToComposition(
        target_col_id="composition", overwrite_data=True
    ).featurize_dataframe(df, "composition")

    if include_oxide_info:
        df = convert_to_oxide_structure(df)
        df = convert_to_oxide_composition(df)

    print(df.describe())
    print(df.head())

    # Write data out to file if user so chooses
    if write_to_csv:
        df.to_csv("elastic_tensor_2018.csv")

    if write_to_compressed_json:
        store_dataframe_as_json(
            df, "elastic_tensor_2018.json.gz", compression="gz"
        )

    return df


if __name__ == "__main__":
    # generate_mp(write_to_csv=True, write_to_compressed_json=True)
    generate_elastic_tensor(write_to_csv=True, write_to_compressed_json=True)
