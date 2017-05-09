from itertools import groupby

import pandas as pd
from tqdm import tqdm

__author__ = 'Anubhav Jain <ajain@lbl.gov>'


class MongoDataRetrieval:

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
            projection: (list) - a list of str fields to grab; dot-notation is allowed.
                Set to "None" to try to auto-detect the fields.
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

        query_proj = [remove_ints(p) for p in clean_projection(projection)]

        print(query_proj)

        r = self.coll.find(query, query_proj, sort=sort)
        if limit:
            r.limit(limit)

        total = min(limit, r.count())

        all_data = []   # matrix of row, column data
        for d in tqdm(r, total=total):
            row_data = []

            # split up dot-notation keys
            for key in projection:
                vals = key.split('.')
                vals = [int(v) if is_int(v) else v for v in vals]
                data = reduce(lambda e, k: e[k], vals, d)
                row_data.append(data)

            all_data.append(row_data)

        df = pd.DataFrame(all_data, columns=projection)
        if idx_field:
            df = df.set_index([idx_field])
        return df


def clean_projection(projection):
    """
    Projecting on e.g. 'a.b.' and 'a' is disallowed. Project inclusively.
    Args:
        projection: (list) - list of fields to grab; dot-notation is allowed.
    """
    return [
        list(g)[0] for _, g in
        groupby(sorted(projection), key=lambda p: p.split(".", 1)[0])]


def remove_ints(projection):
    proj = [p for p in projection.split(".") if not is_int(p)]
    return ".".join(proj)


def is_int(x):
    try:
        int(x)
        return True
    except:
        return False