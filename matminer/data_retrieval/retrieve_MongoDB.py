import pandas as pd
from itertools import groupby
from matminer.data_retrieval.retrieve_base import BaseDataRetrieval
from tqdm import tqdm
from functools import reduce

__author__ = 'Anubhav Jain <ajain@lbl.gov>'


class MongoDataRetrieval(BaseDataRetrieval):
    def __init__(self, coll):
        """
        Retrieves data from a MongoDB collection to a pandas.Dataframe object

        Args:
            coll: A MongoDB collection object
        """
        self.coll = coll

    def api_link(self):
        return "data from\n{}".format(self.coll)

    def get_dataframe(self, criteria, properties=None, limit=0,
                      sort=None, idx_field=None, strict=False):
        """
        Args:
            criteria: (dict) - a pymongo-style query to filter data records
            properties: ([str] or None) - a list of str fields to retrieve;
                dot-notation is allowed (e.g. "structure.lattice.a").
                Set to "None" to try to auto-detect the fields.
            limit: (int) - max number of entries. 0 means no limit
            sort: (tuple) - pymongo-style sort option
            idx_field: (str) - name of field to use as index (must have unique
                entries)
            strict: (bool) - if False, replaces missing values with NaN

        Returns (pandas.DataFrame):

        """
        if properties is None:
            # auto-detect properties/projections as all root keys of a document
            # assumes DB is uniform
            d = self.coll.find_one(criteria, properties, sort=sort)
            properties = d.keys()

        query_proj = [remove_ints(p) for p in clean_projection(properties)]

        r = self.coll.find(criteria, query_proj, sort=sort).limit(limit)

        all_data = []  # matrix of row, column data
        for d in tqdm(r):
            row_data = []

            # split up dot-notation keys
            for key in properties:
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

        df = pd.DataFrame(all_data, columns=properties)
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

    Returns (str)
    """
    proj = [p for p in projection.split(".") if not is_int(p)]
    return ".".join(proj)