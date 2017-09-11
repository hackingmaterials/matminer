import pandas as pd

from pymatgen import MPRester

__author__ = 'Saurabh Bajaj <sbajaj@lbl.gov>, Anubhav Jain <ajain@lbl.gov>'


class MPDataRetrieval:
    """
    MPDataRetrieval is used to retrieve data from the Materials Project
    database, print the results, and convert them into an indexed Pandas
    dataframe.
    """

    def __init__(self, api_key=None):
        """
        Args:
            api_key: (str) Your Materials Project API key, or None if you've
                set up your pymatgen config.
        """

        self.mprester = MPRester(api_key=api_key)

    def get_dataframe(self, criteria, properties, mp_decode=False,
                      index_mpid=True):
        """
        Gets data from MP in a dataframe format.
        See API docs at
        https://materialsproject.org/wiki/index.php/The_Materials_API
        for more details.

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

        Returns: A pandas Dataframe object

        """

        if index_mpid and "material_id" not in properties:
            properties.append("material_id")

        data = self.mprester.query(criteria, properties, mp_decode)
        df = pd.DataFrame(data, columns=properties)

        if index_mpid:
            df = df.set_index("material_id")

        return df
