import pandas as pd
from tqdm import tqdm

__author__ = 'Anubhav Jain <ajain@lbl.gov>'


class MongoDataRetrieval():

    def __init__(self, coll):
        """
        Tool to retrieve data from a MongoDB collection and reformat for data analysis
        Args:
            coll: A MongoDB collection object
        """
        self.coll = coll

    def get_dataframe(self, projection, query=None, sort=None,
                      limit=None, idx_field=None):
        """
        Args:
            projection: (list) - a list of fields to grab; dot-notation is allowed.
                Set to "None" to try to auto-detect the fields
            query: (JSON) - a pymongo-style query to restrict data being gathered
            sort: (tuple) - pymongo-style sort option
            limit: (int) - int to limit the number of entries
            idx_field: (str) - name of field to use as index field (must be unique)
        """
        # auto-detect projection as all root keys of any document
        # assumes DB is uniform
        if not projection:
            d = self.coll.find_one(query, projection, sort=sort)
            projection = d.keys()

        all_data = []   # matrix of row, column data
        r = self.coll.find(query, projection, sort=sort)
        if limit:
            r.limit(limit)

        total = r.count()

        for d in tqdm(r):
            row_data = []

            # split up dot-notation keys
            for key in projection:
                vals = key.split('.')
                data = reduce(lambda d, k: d[k], vals, d)
                row_data.append(data)

            all_data.append(row_data)

        df = pd.DataFrame(all_data, columns=projection)
        if idx_field:
            df = df.set_index([idx_field])
        return df