"""
This module defines functions for writing and reading matminer related objects
"""

import json
import pandas

from monty.io import zopen
from monty.json import MontyEncoder, MontyDecoder


def store_dataframe_as_json(dataframe, filename, compression=None):
    """Store pandas dataframe as a json file.

    Automatically encodes pymatgen objects as dictionaries.

    One downside is that JSON does not support integer dictionary keys.
    Accordingly, if the index of the dataframe is of integer type, it will
    be converted into strings.

    Args:
        dataframe (Pandas.Dataframe): A pandas dataframe.
        filename (str): Path to json file.
        compression (str or None): A compression mode. Valid options are "gz",
            "bz2", and None. Defaults to None. If the filename does not end
            in with the correct suffix it will be added automatically.
    """
    if compression not in ["gz", "bz2", None]:
        raise ValueError("Supported compression formats are 'gz' and 'bz2'.")

    if compression and not filename.lower().endswith(".{}".format(compression)):
        filename = "{}.{}".format(filename, compression)

    write_type = "wb" if compression else "w"

    with zopen(filename, write_type) as f:
        data = json.dumps(dataframe.to_dict(), cls=MontyEncoder)
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
        dataframe_dict = json.load(f, cls=MontyDecoder)
    return pandas.DataFrame.from_dict(dataframe_dict)
