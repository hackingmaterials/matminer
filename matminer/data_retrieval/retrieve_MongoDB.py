import pandas as pd

__author__ = 'Anubhav Jain <ajain@lbl.gov>'


class MongoDataRetrieval():

    def __init__(self, coll, query=None, projection=None, sort=None, limit=None, progressbar=100):
        self.coll = coll
        self.projection = projection

        # initialize data
        # need to know columns of data frame; use a query to determine the fields
        # this assumes the fields are homogeneous
        # TODO: what to do with sub-keys?
        if not self.projection:
            d = self.coll.find_one(query, self.projection, sort=sort)
            self.projection = d.keys()

        self.data = []
        r = self.coll.find(query, self.projection, sort=sort)
        if limit:
            r.limit(limit)

        total = r.count()

        idx=0

        for d in r:
            row_data = []
            for key in self.projection:
                vals = key.split('.')
                data = reduce(lambda d, k: d[k], vals, d)
                row_data.append(data)

            self.data.append(row_data)
            idx += 1
            if idx % progressbar == 0:
                print "{}/{} entries processed...".format(idx, total)

        print 'DONE PRE-PROCESSING'

    def to_pandas(self, idx_field=None):
        df = pd.DataFrame(self.data, columns=self.projection)
        if idx_field:
            df.index = df[idx_field]
        return df