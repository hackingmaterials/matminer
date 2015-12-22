from pymatgen import Composition, Element, MPRester, periodic_table
#from sklearn import linear_model, cross_validation, metrics, ensemble
import pandas as pd
#import matplotlib.pyplot as plt

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

    def to_pandas(self):
        df = pd.DataFrame(self.data, columns=self.properties)
        return df

# TODO: the user has no way to set their API key!! The code will return an error unless the API is set as an environment variable. The MPDataRetrieval should take in the API key as an argument.
# TODO: add an example in an 'examples' package. This exercise would have probably caught some of the errors.
# TODO: make it easy to set the task_id as the default index frame. This should be done in almost all instances unless the user asks not to do it.
# TODO: clean up unused import statements. PyCharm should help you identify what is not needed (greyed out). Others are red underlined (periodic table)