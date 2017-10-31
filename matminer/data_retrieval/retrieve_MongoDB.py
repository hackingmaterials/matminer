from itertools import groupby

import pandas as pd
from tqdm import tqdm

__author__ = 'Anubhav Jain <ajain@lbl.gov>'


class MongoDataRetrieval:
    def __init__(self, coll):
        """
        Tool to retrieve data from a MongoDB collection and put into a pandas
        Dataframe object

        Args:
            coll: A MongoDB collection object
        """
        self.coll = coll

    def get_dataframe(self, projection, query=None, sort=None,
                      limit=None, idx_field=None, strict=False):
        """
        Args:
            projection: (list) - a list of str fields to retrieve; dot-notation is
                allowed. Set to "None" to try to auto-detect the fields.
            query: (JSON) - a pymongo-style query to filter data records
            sort: (tuple) - pymongo-style sort option
            limit: (int) - max number of entries
            idx_field: (str) - name of field to use as index (must have unique
                entries)
            strict: (bool) - if False, replaces missing values with NaN
        """

        if not projection:
            # auto-detect projection as all root keys of any document
            # assumes DB is uniform
            d = self.coll.find_one(query, projection, sort=sort)
            projection = d.keys()

        query_proj = [remove_ints(p) for p in clean_projection(projection)]

        r = self.coll.find(query, query_proj, sort=sort)
        if limit:
            r.limit(limit)

        total = min(limit, r.count())

        all_data = []  # matrix of row, column data
        for d in tqdm(r, total=total):
            row_data = []

            # split up dot-notation keys
            for key in projection:
                try:
                    vals = key.split('.')
                    vals = [int(v) if is_int(v) else v for v in vals]
                    data = reduce(lambda e, k: e[k], vals, d)
                    row_data.append(data)
                except:
                    if not strict:
                        row_data.append(float("nan"))
                    else:
                        raise

            all_data.append(row_data)

        df = pd.DataFrame(all_data, columns=projection)
        if idx_field:
            df = df.set_index([idx_field])
        return df


def clean_projection(projection):
    """
    Projecting on e.g. 'a.b.' and 'a' is disallowed in MongoDb, so project
    inclusively. See unit tests for examples of what this is doing.
    Args:
        projection: (list) - list of fields to retrieve; dot-notation is allowed.
    """
    all_proj = []
    for group in groupby(sorted(projection), key=lambda p: p.split(".", 1)[0]):
        common = ''
        derivs = list(group[1])
        smallest_deriv = derivs[0]
        buffer = ""
        for i in range(len(smallest_deriv)):
            all_match = True
            for deriv in derivs:
                if deriv[i] != smallest_deriv[i]:
                    all_match = False
                    break

            if all_match:
                if smallest_deriv[i] == '.':
                    common += buffer
                    buffer = ''

                buffer += smallest_deriv[i]
                if i == len(smallest_deriv) - 1:
                    common += buffer

            else:
                break

        all_proj.append(common)

    return all_proj

def is_int(x):
    try:
        int(x)
        return True
    except:
        return False

def remove_ints(projection):
    """
    Transforms a string like "a.1.x" to "a.x" - for Mongo projection purposes
    Args:
        projection: (str) the projection to remove ints from

    Returns:

    """

    proj = [p for p in projection.split(".") if not is_int(p)]
    return ".".join(proj)