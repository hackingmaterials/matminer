from functools import reduce

from pandas import DataFrame

from matminer.data_retrieval.retrieve_base import BaseDataRetrieval

from aflow import K  # module of aflow Keyword properties
from aflow.control import Query
from aflow.caster import cast

__author__ = ['Maxwell Dylla <280mtd@gmail.com>']


class AFLOWDataRetrieval(BaseDataRetrieval):
    """Retrieves data from the AFLOW database.

    AFLOW uses the AFLUX API syntax. The aflow library handles the HTTP network
    requests for material properties. Note that this helper library is not an
    official repository of the AFLOW consortium.
    """

    def api_link(self):
        return "https://rosenbrockc.github.io/aflow/index.html"

    def get_dataframe(self, criteria, properties, request_size=10000,
                      request_limit=0, index_auid=True):
        """Retrieves data from AFLOW in a DataFrame format.

        The method builds an AFLUX API query from pymongo-like filter criteria
        and requested properties. Then, results are collected over HTTP.

        Args:
            criteria: (dict) Pymongo-like query operator. The first-level
                dictionary keys must be supported AFLOW properties. The values
                of the dictionary must either be singletons (int, str, etc.) or
                dictionaries. The keys of this second-level dictionary can be
                the pymongo operators '$in', '$gt', '$lt', or '$not.' There can
                not be further nesting.
                VALID:
                    {'auid': {'$in': ['aflow:a17a2da2f3d3953a']}}
                INVALID:
                    {'auid': {'$not': {'$in': ['aflow:a17a2da2f3d3953a']}}}
            properties: (list of str) Properties returned  in the DataFrame.
                See the api link for a list of supported properties.
            request_size: (int) Number of results to return per HTTP request.
            request_limit: (int) Maximum number of requests to submit. The
                default behavior is to request all matching records.
            index_auid: (bool) Whether to set the "AFLOW unique identifier" as
                the index of the DataFrame.

        Returns (pandas.DataFrame): The data requested from the AFLOW database.
        """

        # ensures that 'auid' is in requested properties if desired for index
        if index_auid and ('auid' not in properties):
            properties.append('auid')

        # generates a query for submitting HTTP requests to AFLOW servers
        query = RetrievalQuery.from_pymongo(criteria, properties, request_size)

        # submits HTTP requests and collects results
        df = self.collect_raw_requests(query, request_limit)

        # casts each column into the correct data-type
        for keyword in df.columns.values:
            df[keyword] = self.cast_aflow_series(df[keyword])

        # sets the auid as the index if desired
        if index_auid:
            df.set_index('auid', inplace=True)

        return df

    def collect_raw_requests(self, query, request_limit):
        """Collects the string-casted results of a query.

        Args:
            query: (aflow.control.Query) A query with unprocessed requests.
            request_limit: (int) Maximum number of requests to submit.
        """

        # requests the first page of results to determine number of pages
        query._request(1, query.k)
        page_limit = (query._N // query.k) + 1
        if request_limit and (page_limit > request_limit):
            page_limit = request_limit

        # requests remaining pages
        for page in range(2, page_limit + 1):
            query._request(page, query.k)

        # collects request responses
        records = {}
        for page in range(1, page_limit + 1):
            records.update(query.responses[page])
        return DataFrame.from_dict(data=records, orient='index')

    @staticmethod
    def cast_aflow_series(series):
        """Casts AFLOW data (pandas Series) as the appropriate python type.

        Args:
            series: (pandas.Series) Str data to cast. The name attribute should
                correspond to a string representation of an aflow.Keyword

        Returns: (list) Data casted as the appropriate python object.
        """

        aflow_type, keyword = getattr(K, series.name).atype, series.name
        return [cast(aflow_type, keyword, i) for i in series.values]


class RetrievalQuery(Query):
    """Provides additional methods for constructing class instances.
    """

    @classmethod
    def from_pymongo(cls, criteria, properties, request_size):
        """Generates an aflow Query object from pymongo-like arguments.

        Args:
            criteria: (dict) Pymongo-like query operator. See the
                AFLOWDataRetrieval.get_DataFrame method for more details
            properties: (list of str) Properties returned in the DataFrame.
                See the api link for a list of supported properties.
            request_size: (int) Number of results to return per HTTP request.
        """
        # initializes query
        query = RetrievalQuery(batch_size=request_size)

        # adds filters to query
        query.add_pymongo_filters(criteria)

        # determines properties returned by query
        query.select(*[getattr(K, i) for i in properties])

        # suppresses properties that may have been included in criteria
        # but are not requested properties to be returned
        excluded_keywords = set(criteria.keys()) - set(properties)
        query.exclude(*[getattr(K, i) for i in excluded_keywords])

        return query

    def add_pymongo_filters(self, pymongo_query):
        """Generates aflow filters from a pymongo-like filter.

        Args:
            pymongo_query: (dict) Pymongo-like query operator. See the
                AFLOWDataRetrieval.get_dataframe method for more details
        """

        for str_property, value in pymongo_query.items():

            # converts str representation of property to aflow.Keyword
            keyword = getattr(K, str_property)

            if isinstance(value, dict):  # handles special operators
                for inner_key, inner_value in value.items():

                    if inner_key == '$in':
                        self.filter(
                            reduce(lambda x, y: (x | y),
                                   map(lambda z: (keyword == z), inner_value)))

                    elif inner_key == '$gt':
                        self.filter((keyword > inner_value))

                    elif inner_key == '$lt':
                        self.filter((keyword < inner_value))

                    elif inner_key == '$not':
                        self.filter((~(keyword == inner_value)))

                    else:
                        raise Exception(
                            'Only $in, $gt, $lt, and $not are supported!')

            else:  # handles simple equivalence
                self.filter(keyword == value)


if __name__ == '__main__':
    from aflow.entries import AflowFile

    ret = AFLOWDataRetrieval()
    item = ret.get_dataframe(criteria={'spacegroup_relax': {'$in': [216, 225]},
                                       'natoms': 3,
                                       'enthalpy_formation_atom': {'$lt': 0.0}},
                             properties=['aurl', 'enthalpy_formation_atom',
                                         'positions_fractional', 'geometry',
                                         'files', 'prototype'],
                             request_size=100000, request_limit=12,
                             index_auid=True)
    print(item['positions_fractional'])
