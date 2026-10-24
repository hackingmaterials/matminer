import re

import pandas as pd

from matminer.data_retrieval.retrieve_base import BaseDataRetrieval

# Properties that are not summary fields; they are retrieved per material with
# the get_{prop}_by_material_id methods of MPRester.
BY_MATERIAL_ID_PROPS = [
    "bandstructure",
    "bandstructure_uniform",
    "dos",
    "phonon_dos",
    "phonon_bandstructure",
    "phonon_ddb",
]

__author__ = [
    "Saurabh Bajaj <sbajaj@lbl.gov>",
    "Alireza Faghaninia <alireza.faghaninia@gmail.com>",
    "Anubhav Jain <ajain@lbl.gov>",
]


class MPDataRetrieval(BaseDataRetrieval):
    """
    Retrieves data from the Materials Project database.

    If you use this data retrieval class, please additionally cite:

    Ong, S.P., Cholia, S., Jain, A., Brafman, M., Gunter, D., Ceder, G.,
    Persson, K.A., 2015. The Materials Application Programming Interface
    (API): A simple, flexible and efficient API for materials data based on
    REpresentational State Transfer (REST) principles. Computational
    Materials Science 97, 209–215.
    https://doi.org/10.1016/j.commatsci.2014.10.037

    """

    def __init__(self, api_key=None):
        """
        Args:
            api_key: (str) Your Materials Project API key, or None if you've
                set up your pymatgen config.
        """
        from mp_api.client import MPRester

        if api_key:
            self.mprester = MPRester(api_key=api_key)
        else:
            self.mprester = MPRester()

    def api_link(self):
        return "https://docs.materialsproject.org/downloading-data/using-the-api"

    def get_dataframe(self, criteria, properties, index_mpid=True, **kwargs):
        """
        Gets data from MP in a dataframe format. See api_link for more details.

        Args:
            criteria (str/dict): the same as in get_data
            properties ([str]): the same properties supported as in get_data
                plus: "bandstructure" (line mode), "bandstructure_uniform",
                "dos", "phonon_bandstructure", "phonon_dos". Note that for a
                long list of compounds, it may take a long time to retrieve
                some of these objects.
            index_mpid (bool): the same as in get_data
            kwargs (dict): the same keyword arguments as in get_data

        Returns (pandas.Dataframe):
        """
        fields = [p for p in properties if p not in BY_MATERIAL_ID_PROPS]
        data = self.get_data(criteria=criteria, properties=fields, index_mpid=index_mpid, **kwargs)
        df = pd.DataFrame(data, columns=fields)
        for prop in ["dos", "phonon_dos", "phonon_bandstructure", "phonon_ddb"]:
            if prop in properties:
                df[prop] = self.try_get_prop_by_material_id(prop=prop, material_id_list=df["material_id"].values)
        if "bandstructure" in properties:
            df["bandstructure"] = self.try_get_prop_by_material_id(
                prop="bandstructure",
                material_id_list=df["material_id"].values,
                line_mode=True,
            )
        if "bandstructure_uniform" in properties:
            df["bandstructure_uniform"] = self.try_get_prop_by_material_id(
                prop="bandstructure",
                material_id_list=df["material_id"].values,
                line_mode=False,
            )
        if index_mpid:
            df = df.set_index("material_id")
        return df

    def get_data(self, criteria, properties, mp_decode=True, index_mpid=True):
        """
        Args:
            criteria: (str/dict) the materials to get. A string is a
                material_id ("mp-1234"), a chemical system ("Li-Fe-O") or a
                formula ("Fe2O3", "\\*2O3"). A dict is passed as keyword
                arguments to mp_api's MPRester.materials.summary.search().
                Dict example: {"band_gap": (1, None)}. A "material_id" key is
                also accepted in place of "material_ids".

            properties: (list) the summary fields to get, see
                MPRester.materials.summary.available_fields. Example:
                ["formula_pretty", "formation_energy_per_atom"]

            mp_decode: (bool) Whether to decode to a Pymatgen object where
                possible. If False, these objects are returned as dicts.

            index_mpid: (bool) Whether to set the materials_id as the dataframe
                index.

        Returns ([dict]):
            a list of jsons that match the criteria and contain properties
        """
        if index_mpid and "material_id" not in properties:
            properties.append("material_id")
        docs = self.mprester.materials.summary.search(**self._criteria_to_kwargs(criteria), fields=properties)
        data = []
        for doc in docs:
            entry = {}
            for prop in properties:
                value = getattr(doc, prop, None)
                if prop == "material_id" and value is not None:
                    value = str(value)
                elif not mp_decode and hasattr(value, "as_dict"):
                    value = value.as_dict()
                entry[prop] = value
            data.append(entry)
        return data

    @staticmethod
    def _criteria_to_kwargs(criteria):
        """
        Convert the criteria of get_data to keyword arguments for
        MPRester.materials.summary.search().
        """
        if isinstance(criteria, str):
            if re.fullmatch(r"[a-z]+-\d+", criteria):
                return {"material_ids": [criteria]}
            if "-" in criteria:
                return {"chemsys": criteria}
            return {"formula": criteria}
        kwargs = dict(criteria)
        if "material_id" in kwargs:
            material_ids = kwargs.pop("material_id")
            kwargs["material_ids"] = [material_ids] if isinstance(material_ids, str) else list(material_ids)
        return kwargs

    def try_get_prop_by_material_id(self, prop, material_id_list, **kwargs):
        """
        Call the relevant get_prop_by_material_id. "prop" is a property such
        as bandstructure that is not readily available in supported properties
        of the get_data function but via the get_bandstructure_by_material_id
        method for example.

        Args:
            prop (str): the name of the property. Options are:
                "bandstructure", "dos", "phonon_dos", "phonon_bandstructure",
                "phonon_ddb"
            material_id_list ([str]): list of material_id of compounds
            kwargs (dict): other keyword arguments that get_*_by_material_id
                may have; e.g. line_mode in get_bandstructure_by_material_id

        Returns ([target prop object or NaN]):
            If the target property is not available for a certain material_id,
            NaN is returned.
        """
        from mp_api.client.core import MPRestError

        method = getattr(self.mprester, f"get_{prop}_by_material_id", None)
        if method is None:
            raise ValueError(f"{prop} is not available from the Materials Project API")
        props = []
        for material_id in material_id_list:
            try:
                props.append(method(material_id=material_id, **kwargs))
            except MPRestError:
                props.append(float("NaN"))
        return props

    def citations(self):
        return [
            "@article{Jain2013,"
            "doi = {10.1063/1.4812323},"
            "url = {https://doi.org/10.1063/1.4812323},"
            "year = {2013},"
            "month = jul,"
            "publisher = {{AIP} Publishing},"
            "volume = {1},"
            "number = {1},"
            "pages = {011002},"
            "author = {Anubhav Jain and Shyue Ping Ong and Geoffroy Hautier and Wei "
            "Chen and William Davidson Richards and Stephen Dacek and "
            "Shreyas Cholia and Dan Gunter and David Skinner and "
            "Gerbrand Ceder and Kristin A. Persson},"
            "title = {Commentary: The Materials Project: A materials genome "
            "approach to accelerating materials innovation},"
            "journal = {{APL} Materials}"
            "}",
            "@article{Ong2015,"
            "doi = {10.1016/j.commatsci.2014.10.037},"
            "url = {https://doi.org/10.1016/j.commatsci.2014.10.037},"
            "year = {2015},"
            "month = feb,"
            "publisher = {Elsevier {BV}},"
            "volume = {97},"
            "pages = {209--215},"
            "author = {Shyue Ping Ong and Shreyas Cholia and Anubhav Jain "
            "and Miriam Brafman and Dan Gunter and Gerbrand Ceder and Kristin A. Persson},"
            "title = {The Materials Application Programming Interface ({API}): "
            "A simple,  flexible and efficient {API} for materials data based on "
            "{REpresentational} State Transfer ({REST}) principles},"
            "journal = {Computational Materials Science}"
            "}",
        ]
