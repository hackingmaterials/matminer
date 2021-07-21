"""
This module defines functions for writing and reading matminer related objects
"""

import json
import sys

import pandas
from tqdm import tqdm
from monty.io import zopen
from monty.json import MontyEncoder, MontyDecoder


def store_dataframe_as_json(dataframe, filename, compression=None, orient="split", pbar=True):
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
        pbar (bool): If True, shows a progress bar for encoding objects to
            compatible json format (normally the rate-limiting step).
    """
    if compression not in ["gz", "bz2", None]:
        raise ValueError("Supported compression formats are 'gz' and 'bz2'.")

    if compression and not filename.lower().endswith(".{}".format(compression)):
        filename = "{}.{}".format(filename, compression)

    write_type = "wb" if compression else "w"

    def is_encodable(obj):
        """
        Determine if an object likely is encodable by monty and, consequently,
        will eat compute in encoding.

        Args:
            obj (object): An object which may or may not be converted by monty.

        Returns:
            (bool)
        """
        try:
            m = obj.__module__
            if "pymatgen" in m or "matminer" in m:
                return True
        except AttributeError:
            pass
        return False

    count = 0
    for obj in dataframe.values.flatten():
        if is_encodable(obj):
            count += 1

    pbar1 = tqdm(
        desc=f"Encoding objects into {filename}", position=0, leave=True, ascii=True, disable=not pbar, total=count
    )

    class MontyEncoderPbar(MontyEncoder):
        """
        A pbar-friendly version of MontyEncoder.
        """

        def default(self, o) -> dict:
            if is_encodable(o):
                pbar1.update(1)
            return super().default(o)

    with zopen(filename, write_type) as f:
        data = json.dumps(dataframe.to_dict(orient=orient), cls=MontyEncoderPbar)
        if compression:
            data = data.encode()
        f.write(data)


def load_dataframe_from_json(filename, pbar=True, decode=True):
    """Load pandas dataframe from a json file.

    Automatically decodes and instantiates pymatgen objects in the dataframe.

    Args:
        filename (str): Path to json file. Can be a compressed file (gz and bz2)
            are supported.
        pbar (bool): If true, shows an ASCII progress bar for loading data from disk.
        decode (bool): If true, will automatically decode objects (slow, convenient).
            If false, will return json representations of the objects (fast, inconvenient).

    Returns:
        (Pandas.DataFrame): A pandas dataframe.
    """
    # Progress bar for reading file with hook
    pbar1 = tqdm(desc=f"Reading file {filename}", position=0, leave=True, ascii=True, disable=not pbar)

    def is_monty_object(o):
        """
        Determine if an object can be decoded into json
        by monty.

        Args:
            o (object): An object in dict-form.

        Returns:
            (bool)

        """
        if isinstance(o, dict) and "@class" in o:
            return True
        else:
            return False

    def pbar_hook(obj):
        """
        A hook for a pbar reading the raw data from json, not
        using monty decoding to decode the object.

        Args:
            obj (object): A dict-like

        Returns:
            obj (object)

        """
        if is_monty_object(obj):
            pbar1.update(1)
            sys.stderr.flush()
        return obj

    # Progress bar for decoding objects
    pbar2 = tqdm(desc=f"Decoding objects from {filename}", position=0, leave=True, ascii=True, disable=not pbar)

    class MontyDecoderPbar(MontyDecoder):
        """
        A pbar-friendly version of MontyDecoder.
        """

        def process_decoded(self, d):
            if isinstance(d, dict) and "data" in d and "index" in d and "columns" in d:
                # total number of objects to decode
                # is the number of @class mentions
                pbar2.total = str(d).count("@class")
            elif is_monty_object(d):
                pbar2.update(1)
                sys.stderr.flush()
            return super().process_decoded(d)

    if decode:
        decoder = MontyDecoderPbar if pbar else MontyDecoder
    else:
        decoder = None

    hook = pbar_hook if pbar else lambda x: x

    with zopen(filename, "rb") as f:
        dataframe_data = json.load(f, cls=decoder, object_hook=hook)

    pbar1.close()
    pbar2.close()

    # if only keys are data, columns, index then orient=split
    if isinstance(dataframe_data, dict):
        if set(dataframe_data.keys()) == {"data", "columns", "index"}:
            return pandas.DataFrame(**dataframe_data)
    else:
        return pandas.DataFrame(dataframe_data)
