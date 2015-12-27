from pymatgen import MPRester
import pandas as pd


# TODO: fix the format of the commenting above. use the triple-quote for multi-line comments.


class MPDataRetrieval:
    """
    as
    """

    def __init__(self, criteria, properties, api_key=None, mp_decode=True):
        # TODO: description of args goes here. Use FWS format.

        # api_key (str): A String API key for accessing the MaterialsProject
        # Initializing MPRester. Note that you can call MPRester. MPRester looks for the API key in two places:
        # - Supplying it directly as an __init__ arg.
        # - Setting the "MAPI_KEY" environment variable.
        # Please obtain your API key at https://www.materialsproject.org/dashboard
        self.criteria = criteria
        self.properties = properties
        self.api_key = api_key
        self.mp_decode = mp_decode

        if self.api_key is None:
            mprest = MPRester()
        else:
            mprest = MPRester(self.api_key)

        if "material_id" not in self.properties:
            self.properties.append("material_id")
        self.data = mprest.query(self.criteria, self.properties, self.mp_decode)

    def print_output(self):
        return self.data

    def to_pandas(self, index_mpid=True):
        if index_mpid:
            df = pd.DataFrame(self.data, columns=self.properties)
            indexed_df = df.set_index("material_id")
            return indexed_df
        return pd.DataFrame(self.data, columns=self.properties)

        # TODO: add an example in an 'examples' package. This exercise would have probably caught some of the errors.
