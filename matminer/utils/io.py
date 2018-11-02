"""
This module defines functions for writing and reading matminer related objects
"""

import json
import pandas

from monty.io import zopen
from monty.json import MontyEncoder, MontyDecoder


def store_dataframe_as_json(dataframe, filename, compression=None,
                            orient='split'):
    """Store pandas dataframe as a json file.

    Automatically encodes pymatgen objects as dictionaries.

    Args:
        dataframe (Pandas.Dataframe): A pandas dataframe.
        filename (str): Path to json file.
        compression (str or None): A compression mode. Valid options are "gz",
            "bz2", and None. Defaults to None. If the filename does not end
            in with the correct suffix it will be added automatically.
        orient (str): Determines the format in which the dictionary data is
            stored. This takes the same set of arguments as the `orient` option
            in `pandas.DataFrame.to_dict()` function. 'split' is recommended
            as it is relatively space efficient and preserves the dtype
            of the index.
    """
    if compression not in ["gz", "bz2", None]:
        raise ValueError("Supported compression formats are 'gz' and 'bz2'.")

    if compression and not filename.lower().endswith(".{}".format(compression)):
        filename = "{}.{}".format(filename, compression)

    write_type = "wb" if compression else "w"

    with zopen(filename, write_type) as f:
        data = json.dumps(dataframe.to_dict(orient=orient), cls=MontyEncoder)
        if compression:
            data = data.encode()
        f.write(data)


def load_dataframe_from_json(filename):
    """Load pandas dataframe from a json file.

    Automatically decodes and instantiates pymatgen objects in the dataframe.

    Args:
        filename (str): Path to json file. Can be a compressed file (gz and bz2)
            are supported.

    Returns:
        (Pandas.DataFrame): A pandas dataframe.
    """
    with zopen(filename, 'rb') as f:
        dataframe_data = json.load(f, cls=MontyDecoder)

    # if only keys are data, columns, index then orient=split
    if isinstance(dataframe_data, dict):
        if set(dataframe_data.keys()) == {'data', 'columns', 'index'}:
            return pandas.DataFrame(**dataframe_data)
    else:
        return pandas.DataFrame(dataframe_data)
