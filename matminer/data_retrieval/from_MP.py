from pymatgen import MPRester
import pandas as pd


class MPDataRetrieval:
    """
    MPDataRetrieval is used to retrieve data from the Materials Project database, print the results, and convert them
    into an indexed/unindexed Pandas dataframe.
    """

    def __init__(self, criteria, properties, api_key=None, mp_decode=True):
        """
        :param criteria (str/dict): See MPRester docs for more details.
            Criteria of the query as a string or mongo-style dict.
            If string, it supports a powerful but simple string criteria.
            E.g., "Fe2O3" means search for materials with reduced_formula
            Fe2O3. Wild cards are also supported. E.g., "\*2O" means get
            all materials whose formula can be formed as \*2O, e.g.,
            Li2O, K2O, etc.

            Other syntax examples:
            mp-1234: Interpreted as a Materials ID.
            Fe2O3 or \*2O3: Interpreted as reduced formulas.
            Li-Fe-O or \*-Fe-O: Interpreted as chemical systems.

            You can mix and match with spaces, which are interpreted as
            "OR". E.g. "mp-1234 FeO" means query for all compounds with
            reduced formula FeO or with materials_id mp-1234.

            Using a full dict syntax, even more powerful queries can be
            constructed. For example, {"elements":{"$in":["Li",
            "Na", "K"], "$all": ["O"]}, "nelements":2} selects all Li, Na
            and K oxides. {"band_gap": {"$gt": 1}} selects all materials
            with band gaps greater than 1 eV.
        :param properties (list): See MPRester docs for more details.
            Properties to request for as a list. For example, ["formula",
            "formation_energy_per_atom"] returns the formula and formation energy per atom.
        :param api_key (str): See MPRester docs for more details.
            A String API key for accessing the MaterialsProject. Note that when you call MPRester,
            it looks for the API key in two places:
            - Supplying it directly as an __init__ arg.
            - Setting the "MAPI_KEY" environment variable.
              Please obtain your API key at https://www.materialsproject.org/dashboard
        :param mp_decode (bool): See MPRester docs for more details.
            Whether to do a decoding to a Pymatgen object
            where possible. In some cases, it might be useful to just get
            the raw python dict, i.e., set to False.
        """
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

    def return_output(self):
        """
        :return: results of the query command of pymatgen
        """
        return self.data

    def to_pandas(self, index_mpid=True):
        """
        :param index_mpid (bool): Whether to index the data frame on MPID or not. Defaults to True.
        :return: An indexed or unindexed Pandas data frame of the results.
        """
        self.index_mpid = index_mpid
        if index_mpid:
            df = pd.DataFrame(self.data, columns=self.properties)
            indexed_df = df.set_index("material_id")
            return indexed_df
        return pd.DataFrame(self.data, columns=self.properties)

        # TODO: add an example in an 'examples' package. This exercise would have probably caught some of the errors.

m = MPDataRetrieval(criteria='mp-19717', properties=['cif'])
print m.return_output()
print m.to_pandas()