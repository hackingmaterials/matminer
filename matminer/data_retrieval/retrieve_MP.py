import pandas as pd
from pymatgen.ext.matproj import MPRester, MPRestError

from matminer.data_retrieval.retrieve_base import BaseDataRetrieval

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
        self.mprester = MPRester(api_key=api_key)

    def api_link(self):
        return "https://materialsproject.org/wiki/index.php/The_Materials_API"

    def get_dataframe(self, criteria, properties, index_mpid=True, **kwargs):
        """
        Gets data from MP in a dataframe format. See api_link for more details.

        Args:
            criteria (dict): the same as in get_data
            properties ([str]): the same properties supported as in get_data
                plus: "structure", "initial_structure", "final_structure",
                "bandstructure" (line mode), "bandstructure_uniform",
                "phonon_bandstructure", "phonon_ddb", "phonon_bandstructure",
                "phonon_dos". Note that for a long list of compounds, it may
                take a long time to retrieve some of these objects.
            index_mpid (bool): the same as in get_data
            kwargs (dict): the same keyword arguments as in get_data

        Returns (pandas.Dataframe):
        """
        data = self.get_data(criteria=criteria, properties=properties, index_mpid=index_mpid, **kwargs)
        df = pd.DataFrame(data, columns=properties)
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
            criteria: (str/dict) see MPRester.query() for a description of this
                parameter. String examples: "mp-1234", "Fe2O3", "Li-Fe-O',
                "\\*2O3". Dict example: {"band_gap": {"$gt": 1}}

            properties: (list) see MPRester.query() for a description of this
                parameter. Example: ["formula", "formation_energy_per_atom"]

            mp_decode: (bool) see MPRester.query() for a description of this
                parameter. Whether to decode to a Pymatgen object where
                possible.

            index_mpid: (bool) Whether to set the materials_id as the dataframe
                index.

        Returns ([dict]):
            a list of jsons that match the criteria and contain properties
        """
        if index_mpid and "material_id" not in properties:
            properties.append("material_id")
        data = self.mprester.query(criteria, properties, mp_decode=mp_decode)
        return data

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
        method = getattr(self.mprester, f"get_{prop}_by_material_id")
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
