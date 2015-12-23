from pymatgen import Composition, Element, MPRester
import pandas as pd

# Initializing MPRester. Note that you can call MPRester. MPRester looks for the API key in two places:
# - Supplying it directly as an __init__ arg.
# - Setting the "MAPI_KEY" environment variable.
# Please obtain your API key at https://www.materialsproject.org/dashboard

# TODO: fix the format of the commenting above. use the triple-quote for multi-line comments.

m = MPRester()  # TODO: put this inside the class. You should NOT create global variables like this unless you know what you are doing!

class MPDataRetrieval():
    # TODO: description of class goes here. Use FWS format.

    def __init__(self, criteria, properties, mp_decode=True):
        # TODO: description of args goes here. Use FWS format.

        self.criteria = criteria
        self.properties = properties
        self.mp_decode = mp_decode

        self.data = m.query(criteria, properties, mp_decode=True)

    def print_output(self):
        return self.data

    def to_pandas(self, index_mpid=True):
        if index_mpid:
            if "material_id" not in self.properties:
                self.properties.append("material_id")
                self.data = m.query(self.criteria, self.properties, self.mp_decode)   ## Think about querying again!
            df = pd.DataFrame(self.data, columns=self.properties)
            indexed_df = df.set_index("material_id")
            return indexed_df
        return pd.DataFrame(self.data, columns=self.properties)



# TODO: the user has no way to set their API key!! The code will return an error unless the API is set as an environment variable. The MPDataRetrieval should take in the API key as an argument.
# TODO: add an example in an 'examples' package. This exercise would have probably caught some of the errors.