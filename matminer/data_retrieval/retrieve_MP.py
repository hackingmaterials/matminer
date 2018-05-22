import pandas as pd
from matminer.data_retrieval.retrieve_base import BaseDataRetrieval

from pymatgen import MPRester

__author__ = ['Saurabh Bajaj <sbajaj@lbl.gov>',
             'Alireza Faghaninia <alireza.faghaninia@gmail.com>',
             'Anubhav Jain <ajain@lbl.gov>']


class MPDataRetrieval(BaseDataRetrieval):
    """
    Retrieves data from the Materials Project database.
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
            all arguments including criteria, properties and index_mpid are the
            same as in get_data

        Returns (pandas.Dataframe):
        """
        data = self.get_data(criteria=criteria, properties=properties,
                             index_mpid=index_mpid, **kwargs)
        df = pd.DataFrame(data, columns=properties)
        if index_mpid:
            df = df.set_index("material_id")
        return df

    def get_data(self, criteria, properties, mp_decode=False, index_mpid=True):
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
        data = self.mprester.query(criteria, properties, mp_decode)
        return data