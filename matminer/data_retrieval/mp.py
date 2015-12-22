from pymatgen import Composition, Element, MPRester, periodic_table
#from sklearn import linear_model, cross_validation, metrics, ensemble
import pandas as pd
#import matplotlib.pyplot as plt

# Initializing MPRester. Note that you can call MPRester. MPRester looks for the API key in two places:
# - Supplying it directly as an __init__ arg.
# - Setting the "MAPI_KEY" environment variable.
# Please obtain your API key at https://www.materialsproject.org/dashboard

m = MPRester()

class MPDataRetrieval():

    def __init__(self, criteria, properties, mp_decode=True):
        self.criteria = criteria
        self.properties = properties
        self.mp_decode = mp_decode

        self.data = m.query(criteria, properties, mp_decode=True)

    def print_output(self):
        return self.data

    def to_pandas(self):
        df = pd.DataFrame(self.data, columns=self.properties)
        return df



