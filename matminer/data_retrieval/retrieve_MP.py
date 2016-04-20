import pandas as pd

__author__ = 'Saurabh Bajaj <sbajaj@lbl.gov>'


class MPDataRetrieval:
    """
    MPDataRetrieval is used to retrieve data from the Materials Project database, print the results, and convert them
    into an indexed/unindexed Pandas dataframe.
    """

    def __init__(self, mprester):
        """
        :param mprester (MPRester): A pymatgen MPRester object. See MPRester docs for more details.
        """
        self.mprester = mprester

    def get_dataframe(self, criteria, properties, mp_decode=False, index_mpid=True):
        """
        :param criteria: (str/dict) See MPRester docs for more details.
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
        :param properties: (list) See MPRester docs for more details.
            Properties to request for as a list. For example, ["formula",
            "formation_energy_per_atom"] returns the formula and formation energy per atom.
        :param mp_decode: (bool) See MPRester docs for more details.
            Whether to do a decoding to a Pymatgen object
            where possible. In some cases, it might be useful to just get
            the raw python dict, i.e., set to False.
        :param index_mpid: (bool) Whether to set the materials_id as the dataframe index
        """

        if index_mpid and "material_id" not in properties:
            properties.append("material_id")

        data = self.mprester.query(criteria, properties, mp_decode)
        df = pd.DataFrame(data, columns=properties)

        if index_mpid:
            df = df.set_index("material_id")

        return df
