from matminer.datasets import load_dataset


"Convenience functions provided to make accessing datasets simpler"


def load_elastic_tensor(version="2015", include_metadata=False, data_home=None,
                        download_if_missing=True):
    """
    Convenience function for loading the elastic_tensor dataset.

    Args:
        version (str): Version of the elastic_tensor dataset to load
            (defaults to 2015)

        include_metadata (bool): Whether or not to include the cif, meta,
            and poscar dataset columns. False by default.

        data_home (str, None): Where to loom for and store the loaded dataset

        download_if_missing (bool): Whether or not to download the dataset if
            it isn't on disk

    Returns: (pd.DataFrame)
    """
    df = load_dataset("elastic_tensor" + "_" + version, data_home,
                      download_if_missing)

    if not include_metadata:
        df = df.drop(['cif', 'kpoint_density', 'poscar'], axis=1)

    return df


def load_piezoelectric_tensor(include_metadata=False, data_home=None,
                              download_if_missing=True):
    """
    Convenience function for loading the piezoelectric_tensor dataset.

    Args:
        include_metadata (bool): Whether or not to include the cif, meta,
            and poscar dataset columns. False by default.

        data_home (str, None): Where to loom for and store the loaded dataset

        download_if_missing (bool): Whether or not to download the dataset if
            it isn't on disk

    Returns: (pd.DataFrame)
    """
    df = load_dataset("piezoelectric_tensor", data_home, download_if_missing)

    if not include_metadata:
        df = df.drop(['cif', 'meta', 'poscar'], axis=1)

    return df


def load_dielectric_constant(include_metadata=False, data_home=None,
                             download_if_missing=True):
    """
    Convenience function for loading the dielectric_constant dataset.

    Args:
        include_metadata (bool): Whether or not to include the cif, meta,
            and poscar dataset columns. False by default.

        data_home (str, None): Where to loom for and store the loaded dataset

        download_if_missing (bool): Whether or not to download the dataset if
            it isn't on disk

    Returns: (pd.DataFrame)
    """
    df = load_dataset("dielectric_constant", data_home, download_if_missing)

    if not include_metadata:
        df = df.drop(['cif', 'meta', 'poscar'], axis=1)

    return df


def load_flla(data_home=None, download_if_missing=True):
    """
    Convenience function for loading the flla dataset.

    Args:
        data_home (str, None): Where to loom for and store the loaded dataset

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
        data_home (str, None): Where to loom for and store the loaded dataset

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
        data_home (str, None): Where to loom for and store the loaded dataset

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
        data_home (str, None): Where to loom for and store the loaded dataset

        download_if_missing (bool): Whether or not to download the dataset if
            it isn't on disk

    Returns: (pd.DataFrame)
    """
    df = load_dataset("phonon_dielectric_mp", data_home, download_if_missing)

    return df


def load_glass_ternary_landolt(processing="all", unique_composition=True,
                               data_home=None, download_if_missing=True):
    """
    Convenience function for loading the glass_ternary_landolt dataset.

    Args:
        processing (str): return only items with a specified processing method
            defaults to all, options are sputtering and meltspin

        unique_composition (bool): Whether or not to combine compositions with
            the same formula

        data_home (str, None): Where to loom for and store the loaded dataset

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


def load_double_perovskites_gap(return_lumo=False, data_home=None,
                                download_if_missing=True):
    """
    Convenience function for loading the double_perovskites_gap dataset.

    Args:
        return_lumo (bool) Whether or not to provide LUMO energy dataframe in
            addition to gap dataframe. Defaults to False.

        data_home (str, None): Where to loom for and store the loaded dataset

        download_if_missing (bool): Whether or not to download the dataset if
            it isn't on disk

    Returns: (pd.DataFrame, tuple)
    """
    df = load_dataset("double_perovskites_gap")

    if return_lumo:
        lumo = load_dataset("double_perovskites_gap_lumo", data_home,
                            download_if_missing)
        return df, lumo

    return df


def load_double_perovskites_gap_lumo(data_home=None, download_if_missing=True):
    """
    Convenience function for loading the double_perovskites_gap_lumo dataset.

    Args:
        data_home (str, None): Where to loom for and store the loaded dataset

        download_if_missing (bool): Whether or not to download the dataset if
            it isn't on disk

    Returns: (pd.DataFrame)
    """
    df = load_dataset("double_perovskites_gap_lumo",
                      data_home, download_if_missing)

    return df


def load_glass_ternary_hipt(system="all", data_home=None,
                            download_if_missing=True):
    """
    Convenience function for loading the glass_ternary_hipt dataset.

    Args:
        system (str, list): return items only from the requested system(s)
            options are: "CoFeZr", "CoTiZr", "CoVZr", "FeTiNb"

        data_home (str, None): Where to loom for and store the loaded dataset

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
                raise AttributeError("some of the system list {} are not "
                                     "in this dataset". format(system))
        df = df[df["system"].isin(system)]

    return df


def load_citrine_thermal_conductivity(room_temperature=True, data_home=None,
                                      download_if_missing=True):
    """
    Convenience function for loading the citrine thermal conductivity dataset.

    Args:
        room_temperature (bool) Whether or not to only return items with room
            temperature k_condition. True by default.

        data_home (str, None): Where to loom for and store the loaded dataset

        download_if_missing (bool): Whether or not to download the dataset if
           it isn't on disk

    Returns: (pd.DataFrame)
    """
    df = load_dataset("citrine_thermal_conductivity", data_home,
                      download_if_missing)

    if room_temperature:
        df = df[df['k_condition'].isin(['room temperature',
                                        'Room temperature',
                                        'Standard',
                                        '298', '300'])]
    return df.drop(['k-units', 'k_condition', 'k_condition_units'], axis=1)


def load_mp(include_structures=False, data_home=None, download_if_missing=True):
    """
    Convenience function for loading the materials project dataset.

    Args:
        include_structures (bool) Whether or not to load the full mp
            structure data. False by default.

        data_home (str, None): Where to loom for and store the loaded dataset

        download_if_missing (bool): Whether or not to download the dataset if
           it isn't on disk

    Returns: (pd.DataFrame)
    """
    if include_structures:
        df = load_dataset('mp_all', data_home, download_if_missing)
    else:
        df = load_dataset('mp_nostruct', data_home, download_if_missing)

    return df


def load_wolverton_oxides(data_home=None, download_if_missing=True):
    """
    Convenience function for loading the wolverton oxides dataset.

    Args:
        data_home (str, None): Where to loom for and store the loaded dataset

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
        data_home (str, None): Where to loom for and store the loaded dataset

        download_if_missing (bool): Whether or not to download the dataset if
           it isn't on disk

    Returns: (pd.DataFrame)
    """
    df = load_dataset("heusler_magnetic", data_home, download_if_missing)

    return df

