import pandas as pd

from matminer.datasets import load_dataset

# Convenience functions provided to make accessing datasets simpler.

__author__ = "Daniel Dopp <dbdopp@lbl.gov>" "Abhinav Ashar <abhinav_ashar@berkeley.edu>"


def load_elastic_tensor(version="2015", include_metadata=False, data_home=None, download_if_missing=True):
    """
    Convenience function for loading the elastic_tensor dataset.

    Args:
        version (str): Version of the elastic_tensor dataset to load
            (defaults to 2015)

        include_metadata (bool): Whether or not to include the cif, meta,
            and poscar dataset columns. False by default.

        data_home (str, None): Where to look for and store the loaded dataset

        download_if_missing (bool): Whether or not to download the dataset if
            it isn't on disk

    Returns: (pd.DataFrame)
    """
    df = load_dataset("elastic_tensor" + "_" + version, data_home, download_if_missing)

    if not include_metadata:
        df = df.drop(["cif", "kpoint_density", "poscar"], axis=1)

    return df


def load_piezoelectric_tensor(include_metadata=False, data_home=None, download_if_missing=True):
    """
    Convenience function for loading the piezoelectric_tensor dataset.

    Args:
        include_metadata (bool): Whether or not to include the cif, meta,
            and poscar dataset columns. False by default.

        data_home (str, None): Where to look for and store the loaded dataset

        download_if_missing (bool): Whether or not to download the dataset if
            it isn't on disk

    Returns: (pd.DataFrame)
    """
    df = load_dataset("piezoelectric_tensor", data_home, download_if_missing)

    if not include_metadata:
        df = df.drop(["cif", "meta", "poscar"], axis=1)

    return df


def load_dielectric_constant(include_metadata=False, data_home=None, download_if_missing=True):
    """
    Convenience function for loading the dielectric_constant dataset.

    Args:
        include_metadata (bool): Whether or not to include the cif, meta,
            and poscar dataset columns. False by default.

        data_home (str, None): Where to look for and store the loaded dataset

        download_if_missing (bool): Whether or not to download the dataset if
            it isn't on disk

    Returns: (pd.DataFrame)
    """
    df = load_dataset("dielectric_constant", data_home, download_if_missing)

    if not include_metadata:
        df = df.drop(["cif", "meta", "poscar"], axis=1)

    return df


def load_flla(data_home=None, download_if_missing=True):
    """
    Convenience function for loading the flla dataset.

    Args:
        data_home (str, None): Where to look for and store the loaded dataset

        download_if_missing (bool): Whether or not to download the dataset if
            it isn't on disk

    Returns: (pd.DataFrame)
    """
    df = load_dataset("flla", data_home, download_if_missing)

    return df


def load_castelli_perovskites(data_home=None, download_if_missing=True):
    """
    Convenience function for loading the castelli_perovskites dataset.

    Args:
        data_home (str, None): Where to look for and store the loaded dataset

        download_if_missing (bool): Whether or not to download the dataset if
            it isn't on disk

    Returns: (pd.DataFrame)
    """
    df = load_dataset("castelli_perovskites", data_home, download_if_missing)

    return df


def load_boltztrap_mp(data_home=None, download_if_missing=True):
    """
    Convenience function for loading the boltztrap_mp dataset.

    Args:
        data_home (str, None): Where to look for and store the loaded dataset

        download_if_missing (bool): Whether or not to download the dataset if
            it isn't on disk

    Returns: (pd.DataFrame)
    """
    df = load_dataset("boltztrap_mp", data_home, download_if_missing)

    return df


def load_phonon_dielectric_mp(data_home=None, download_if_missing=True):
    """
    Convenience function for loading the phonon_dielectric_mp dataset.

    Args:
        data_home (str, None): Where to look for and store the loaded dataset

        download_if_missing (bool): Whether or not to download the dataset if
            it isn't on disk

    Returns: (pd.DataFrame)
    """
    df = load_dataset("phonon_dielectric_mp", data_home, download_if_missing)

    return df


def load_glass_ternary_landolt(processing="all", unique_composition=True, data_home=None, download_if_missing=True):
    """
    Convenience function for loading the glass_ternary_landolt dataset.

    Args:
        processing (str): return only items with a specified processing method
            defaults to all, options are sputtering and meltspin

        unique_composition (bool): Whether or not to combine compositions with
            the same formula

        data_home (str, None): Where to look for and store the loaded dataset

        download_if_missing (bool): Whether or not to download the dataset if
            it isn't on disk

    Returns: (pd.DataFrame)
    """
    df = load_dataset("glass_ternary_landolt", data_home, download_if_missing)

    if processing != "all":
        if processing in {"meltspin", "sputtering"}:
            df = df[df["processing"] == processing]

        else:
            raise ValueError("Error, processing method unrecognized")

    if unique_composition:
        df = df.groupby("formula").max().reset_index()

    return df


def load_double_perovskites_gap(return_lumo=False, data_home=None, download_if_missing=True):
    """
    Convenience function for loading the double_perovskites_gap dataset.

    Args:
        return_lumo (bool) Whether or not to provide LUMO energy dataframe in
            addition to gap dataframe. Defaults to False.

        data_home (str, None): Where to look for and store the loaded dataset

        download_if_missing (bool): Whether or not to download the dataset if
            it isn't on disk

    Returns: (pd.DataFrame, tuple)
    """
    df = load_dataset("double_perovskites_gap")

    if return_lumo:
        lumo = load_dataset("double_perovskites_gap_lumo", data_home, download_if_missing)
        return df, lumo

    return df


def load_double_perovskites_gap_lumo(data_home=None, download_if_missing=True):
    """
    Convenience function for loading the double_perovskites_gap_lumo dataset.

    Args:
        data_home (str, None): Where to look for and store the loaded dataset

        download_if_missing (bool): Whether or not to download the dataset if
            it isn't on disk

    Returns: (pd.DataFrame)
    """
    df = load_dataset("double_perovskites_gap_lumo", data_home, download_if_missing)

    return df


def load_glass_ternary_hipt(system="all", data_home=None, download_if_missing=True):
    """
    Convenience function for loading the glass_ternary_hipt dataset.

    Args:
        system (str, list): return items only from the requested system(s)
            options are: "CoFeZr", "CoTiZr", "CoVZr", "FeTiNb"

        data_home (str, None): Where to look for and store the loaded dataset

        download_if_missing (bool): Whether or not to download the dataset if
            it isn't on disk

    Returns: (pd.DataFrame)
    """
    df = load_dataset("glass_ternary_hipt", data_home, download_if_missing)

    if system != "all":
        if isinstance(system, str):
            system = [system]

        for item in system:
            if item not in {"CoFeZr", "CoTiZr", "CoVZr", "FeTiNb"}:
                raise AttributeError("some of the system list {} are not " "in this dataset".format(system))
        df = df[df["system"].isin(system)]

    return df


def load_citrine_thermal_conductivity(room_temperature=True, data_home=None, download_if_missing=True):
    """
    Convenience function for loading the citrine thermal conductivity dataset.

    Args:
        room_temperature (bool) Whether or not to only return items with room
            temperature k_condition. True by default.

        data_home (str, None): Where to look for and store the loaded dataset

        download_if_missing (bool): Whether or not to download the dataset if
           it isn't on disk

    Returns: (pd.DataFrame)
    """
    df = load_dataset("citrine_thermal_conductivity", data_home, download_if_missing)

    if room_temperature:
        df = df[df["k_condition"].isin(["room temperature", "Room temperature", "Standard", "298", "300"])]
    return df.drop(["k-units", "k_condition", "k_condition_units"], axis=1)


def load_mp(include_structures=False, data_home=None, download_if_missing=True):
    """
    Convenience function for loading the materials project dataset.

    Args:
        include_structures (bool) Whether or not to load the full mp
            structure data. False by default.

        data_home (str, None): Where to look for and store the loaded dataset

        download_if_missing (bool): Whether or not to download the dataset if
           it isn't on disk

    Returns: (pd.DataFrame)
    """
    if include_structures:
        df = load_dataset("mp_all_20181018", data_home, download_if_missing)
    else:
        df = load_dataset("mp_nostruct_20181018", data_home, download_if_missing)

    return df


def load_wolverton_oxides(data_home=None, download_if_missing=True):
    """
    Convenience function for loading the wolverton oxides dataset.

    Args:
        data_home (str, None): Where to look for and store the loaded dataset

        download_if_missing (bool): Whether or not to download the dataset if
           it isn't on disk

    Returns: (pd.DataFrame)
    """
    df = load_dataset("wolverton_oxides", data_home, download_if_missing)

    return df


def load_heusler_magnetic(data_home=None, download_if_missing=True):
    """
    Convenience function for loading the heusler magnetic dataset.

    Args:
        data_home (str, None): Where to look for and store the loaded dataset

        download_if_missing (bool): Whether or not to download the dataset if
           it isn't on disk

    Returns: (pd.DataFrame)
    """
    df = load_dataset("heusler_magnetic", data_home, download_if_missing)

    return df


def load_steel_strength(data_home=None, download_if_missing=True):
    """
    Convenience function for loading the steel strength dataset.

    Args:
        data_home (str, None): Where to look for and store the loaded dataset

        download_if_missing (bool): Whether or not to download the dataset if
           it isn't on disk

    Returns: (pd.DataFrame)
    """
    df = load_dataset("steel_strength", data_home, download_if_missing)

    return df


def load_jarvis_ml_dft_training(drop_nan_columns=None, data_home=None, download_if_missing=True):
    """
    Convenience function for loading the jarvis ml dft training dataset.

    Args:
        drop_nan_columns (list, str): Column or columns to drop rows
        containing NaN values from

        data_home (str, None): Where to look for and store the loaded dataset

        download_if_missing (bool): Whether or not to download the dataset if
           it isn't on disk

    Returns: (pd.DataFrame)
    """
    df = load_dataset("jarvis_ml_dft_training", data_home, download_if_missing)

    if drop_nan_columns is None:
        drop_nan_columns = []
    elif isinstance(drop_nan_columns, str):
        drop_nan_columns = [drop_nan_columns]

    return df.dropna(subset=drop_nan_columns)


def load_jarvis_dft_3d(drop_nan_columns=None, data_home=None, download_if_missing=True):
    """
    Convenience function for loading the jarvis dft 3d dataset.

    Args:
        drop_nan_columns (list, str): Column or columns to drop rows
        containing NaN values from

        data_home (str, None): Where to look for and store the loaded dataset

        download_if_missing (bool): Whether or not to download the dataset if
           it isn't on disk

    Returns: (pd.DataFrame)
    """
    df = load_dataset("jarvis_dft_3d", data_home, download_if_missing)

    if drop_nan_columns is None:
        drop_nan_columns = []
    elif isinstance(drop_nan_columns, str):
        drop_nan_columns = [drop_nan_columns]

    return df.dropna(subset=drop_nan_columns)


def load_jarvis_dft_2d(drop_nan_columns=None, data_home=None, download_if_missing=True):
    """
    Convenience function for loading the jarvis dft 2d dataset.

    Args:
        drop_nan_columns (list, str): Column or columns to drop rows
        containing NaN values from

        data_home (str, None): Where to look for and store the loaded dataset

        download_if_missing (bool): Whether or not to download the dataset if
           it isn't on disk

    Returns: (pd.DataFrame)
    """
    df = load_dataset("jarvis_dft_2d", data_home, download_if_missing)

    if drop_nan_columns is None:
        drop_nan_columns = []
    elif isinstance(drop_nan_columns, str):
        drop_nan_columns = [drop_nan_columns]

    return df.dropna(subset=drop_nan_columns)


def load_glass_binary(version="v2", data_home=None, download_if_missing=True):
    """
    Convenience function for loading the glass_binary dataset.

    Args:
        version (str): Version identifier for dataset, see dataset description
            for explanation of each. Defaults to v2

        data_home (str, None): Where to look for and store the loaded dataset

        download_if_missing (bool): Whether or not to download the dataset if
            it isn't on disk

    Returns: (pd.DataFrame)
    """

    # Set version tag if dataset is updated to a new version
    dataset_identifier = "glass_binary"
    if version != "v1":
        dataset_identifier = "_".join([dataset_identifier, version])

    df = load_dataset(dataset_identifier, data_home, download_if_missing)

    return df


def load_m2ax(data_home=None, download_if_missing=True):
    """
    Convenience function for loading the m2ax dataset.

    Args:
        data_home (str, None): Where to look for and store the loaded dataset

        download_if_missing (bool): Whether or not to download the dataset if
            it isn't on disk

    Returns: (pd.DataFrame)
    """
    df = load_dataset("m2ax", data_home, download_if_missing)

    return df


def load_expt_gap(data_home=None, download_if_missing=True):
    """
    Convenience function for loading the expt_gap dataset.me

    Args:
        data_home (str, None): Where to look for and store the loaded dataset

        download_if_missing (bool): Whether or not to download the dataset if
            it isn't on disk

    Returns: (pd.DataFrame)
    """
    df = load_dataset("expt_gap", data_home, download_if_missing)

    return df


def load_expt_formation_enthalpy(data_home=None, download_if_missing=True):
    """
    Convenience function for loading the expt_formation_enthalpy dataset.

    Args:
        data_home (str, None): Where to look for and store the loaded dataset

        download_if_missing (bool): Whether or not to download the dataset if
            it isn't on disk

    Returns: (pd.DataFrame)
    """
    df = load_dataset("expt_formation_enthalpy", data_home, download_if_missing)

    return df


def load_brgoch_superhard_training(subset="all", drop_suspect=False, data_home=None, download_if_missing=True):
    """
    Convenience function for loading the expt_formation_enthalpy dataset.

    Args:
        subset (str): Identifier for subset of data to return,
            all: all possible columns including metadata, engineered features,
                 and basic descriptors
            brgoch_features: only features from reference paper and targets
            basic_descriptors: only composition/structure columns and targets

        drop_suspect (bool): Whether to drop values with possibly incorrect
            elastic data and materials that could not be verified

        data_home (str, None): Where to look for and store the loaded dataset

        download_if_missing (bool): Whether or not to download the dataset if
            it isn't on disk

    Returns: (pd.DataFrame)
    """
    if subset not in {"all", "brgoch_features", "basic_descriptors"}:
        raise ValueError("Error: dataset subset identifier {} " "not recognized".format(subset))

    df = load_dataset("brgoch_superhard_training", data_home, download_if_missing)

    if drop_suspect:
        df = df[~df["suspect_value"]]

    if subset in {"all", "brgoch_features"}:
        feats_expanded = pd.DataFrame([feat_dict for feat_dict in df["brgoch_feats"]])

        for column in feats_expanded.columns:
            df[column] = feats_expanded[column]

    if subset == "basic_descriptors":
        df = df.drop(
            [feat for feat in df.columns if feat not in {"composition", "structure", "shear_modulus", "bulk_modulus"}],
            axis=1,
        )
    elif subset == "brgoch_features":
        df = df.drop(
            [
                "composition",
                "structure",
                "formula",
                "material_id",
                "suspect_value",
                "brgoch_feats",
            ],
            axis=1,
        )
    return df
