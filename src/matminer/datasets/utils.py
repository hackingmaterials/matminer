import hashlib
import json
import os
import time
import warnings

import pandas as pd
import requests
import tqdm

__author__ = "Daniel Dopp <dbdopp@lbl.gov>"


def _load_dataset_dict():
    """
    Loads the dataset dictionary from a storage file

    Returns: (dict)
    """
    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_metadata.json"), encoding="utf-8"
    ) as infile:
        dataset_dict = json.load(infile)

    return dataset_dict


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
        data_home = os.environ.get("MATMINER_DATA", os.path.dirname(os.path.abspath(__file__)))

    data_home = os.path.expanduser(data_home)

    return data_home


def _validate_dataset(data_path, url=None, file_hash=None, download_if_missing=True, n_retries_allowed=3):
    """
    Checks to see if a dataset is on the local machine,
    if not tries to download if download_if_missing is set to true,
    also checks that the hash of the file data matches that included in the
    metadata

    Args:
        data_path (str): the full path to the file you would like to load,
        if nonexistent will try to download from external source by default

        url (str, None): a string specifying the url to fetch the dataset from
        if it is not available

        file_hash (str, None): hash of file used to check for file integrity

        download_if_missing (bool): whether or not to try downloading the
        dataset if it is not on local disk

        n_retries_allowed (int): Number of retries to do before failing a dataset
            download based on hash mismatch. Retries are spaced apart by
            60s.

    Returns (None)
    """
    download_retry_wait = 60

    if n_retries_allowed < 0:
        raise ValueError("Number of retries for download cannot be less than 0.")

    do_download = False
    # If the file doesn't exist, download it
    if not os.path.exists(data_path):
        # Ensure proper arguments for download
        if not download_if_missing:
            raise OSError("Data not found and download_if_missing set to False")
        elif url is None:
            raise ValueError("To download an external dataset, the url metadata must be provided")

        # Ensure storage location exists
        data_home = os.path.dirname(data_path)

        if not os.path.exists(data_home):
            print(f"Making dataset storage folder at {data_home}", flush=True)
            os.makedirs(data_home)

        do_download = True

    hash_mismatch_msg = (
        "Error, hash of downloaded file does not match that " "included in metadata, the data may be corrupt or altered"
    )
    if do_download:
        n_retries = 0
        while n_retries <= n_retries_allowed:
            try:
                _fetch_external_dataset(url, data_path)

                # Check to see if file hash matches the expected value, if hash is provided
                if file_hash is not None:
                    if file_hash != _get_file_sha256_hash(data_path):
                        raise UserWarning
                break
            except UserWarning:
                warnings.warn(hash_mismatch_msg)
                if n_retries < n_retries_allowed:
                    warnings.warn(f"Waiting {download_retry_wait}s and trying again...")
                    time.sleep(download_retry_wait)
                else:
                    raise UserWarning(
                        f"File could not be downloaded to {data_path} after {n_retries_allowed} retries"
                        f"due to repeated hash validation failures."
                    )
                if os.path.exists(data_path):
                    os.remove(data_path)
                n_retries += 1


def _fetch_external_dataset(url, file_path):
    """
    Downloads file from a given url

    Args:
        url (str): string of where to get external dataset

        file_path (str): string of where to save the file to be retrieved

    Returns (None)
    """

    # Fetch data from given url
    msg = f"Fetching {os.path.basename(file_path)} from {url} to {file_path}"
    print(msg, flush=True)

    r = requests.get(url, stream=True)

    pbar = tqdm.tqdm(
        desc=f"Fetching {url} in MB",
        position=0,
        leave=True,
        ascii=True,
        total=len(r.content),
        unit="MB",
        unit_scale=1e-6,
    )
    chunk_size = 2048
    with open(file_path, "wb") as file_out:
        for chunk in r.iter_content(chunk_size=chunk_size):
            pbar.update(chunk_size)
            file_out.write(chunk)

    r.close()


def _get_file_sha256_hash(file_path):
    """
    Takes a file and returns the SHA 256 hash of its data

    Args:
        file_path (str): path of file to hash

    Returns: (str)

    """
    sha256hash = hashlib.sha256()
    chunk_size = 8192
    with open(file_path, "rb") as f:
        while True:
            buffer = f.read(chunk_size)
            if not buffer:
                break
            sha256hash.update(buffer)
    return sha256hash.hexdigest()


def _read_dataframe_from_file(file_path, **kwargs):
    """
    Reads a dataset from a generic file type into a pandas dataframe

    Args:
        file_path (str): path to dataset

        kwargs: arbitrary keywords for any given read_ function in pandas

    Returns: (pd.DataFrame)
    """
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path, **kwargs)
    elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
        df = pd.read_excel(file_path, **kwargs)
    elif file_path.endswith(".json"):
        df = pd.read_json(file_path, **kwargs)
    else:
        raise ValueError(f"File type of {file_path} unsupported")

    return df
