from time import sleep

import pandas as pd
from tqdm import tqdm

from matminer.utils.io import store_dataframe_as_json
from matminer.data_retrieval.retrieve_MP import MPDataRetrieval, MPRestError


# Functions for generating static datasets from various database resources


def generate_mp(max_nsites=None, properties=None, write_to_csv=False,
                write_to_compressed_json=True):
    """
    Grabs all mp materials. This will return two csv/json.gz files:
        * mp_nostruct: All MP materials, not including structures
        * mp_all: All MP materials, including structures

    Args:
        max_nsites (int): The maximum number of sites to include in the query.

        properties (iterable of strings): list of properties supported by
            MPDataRetrieval

        write_to_csv (bool): whether to write resulting dataframe to csv

        write_to_compressed_json (bool): whether to write resulting
            dataframe to json.gz file

    Returns (pandas.DataFrame):
        retrieved/generated data
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


if __name__ == "__main__":
    generate_mp()
