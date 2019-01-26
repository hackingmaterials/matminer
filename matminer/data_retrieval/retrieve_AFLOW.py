from functools import reduce

from pandas import DataFrame

from matminer.data_retrieval.retrieve_base import BaseDataRetrieval

from aflow import K  # module of aflow Keyword properties
from aflow.control import Query

__author__ = ['Maxwell Dylla <280mtd@gmail.com>']


class AFLOWDataRetrieval(BaseDataRetrieval):
    """Retrieves data from the AFLOW database.

    AFLOW uses the AFLUX API syntax. The aflow library handles the HTTP network
    requests for material properties. Note that this helper library is not an
    offical reposiotry of the AFLOW consortium.
    """

    def api_link(self):
        return "https://rosenbrockc.github.io/aflow/index.html"

    def get_dataframe(self, criteria, properties, request_size=10000,
                      request_limit=0, index_auid=True, autocast=False):
        """Retrieves data from AFLOW in a dataframe format.

        The method builds an AFLUX API query from the given filter criteria
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
            properties: (list of str) Properties returned  in the dataframe.
                See the api link for a list of supported properties.
            request_size: (int) Number of results to return per HTTP request.
            request_limit: (int) Maximum number of requests to submit. The
                default behavior is to request all matching records.
            index_auid: (bool) Whether to set the "AFLOW unique identifier" as
                the index of the dataframe.
            autocast: (bool) Whether to autocast the types of each column using
                the aflow library. If no autocasting, then all values are cast
                as strings. There is a large speed-penalty for autocasting.

        Returns (pandas.Dataframe): The data requested from the AFLOW database.
        """

        # ensures that 'auid' is in requested properties if desired for index
        if index_auid and ('auid' not in properties):
            properties.append('auid')

        # generates a query for submitting HTTP requests to AFLOW servers
        query = RetrievalQuery.from_pymongo(criteria, properties, request_size)

        # submits HTTP requests and collects results
        if autocast:  # type casts each result (slow)
            df = self.collect_casted_requests(query, request_limit, properties)
        else:  # bypasses type casting (fast)
            df = self.collect_raw_requests(query, request_limit)

        # sets the auid as the index if desired
        if index_auid:
            df.set_index('auid', inplace=True)

        return df

    def collect_raw_requests(self, query, request_limit):
        """Collects the string-casted results of a query (fast).

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

    def collect_casted_requests(self, query, request_limit, properties):
        """Collects the type-casted results of a query (slow).

        aflow.control.Query supports lazy evaluation over the results of the
        query. This is slow, but also offers easy casing of the properties.

        Args:
            query: (aflow.control.Query) A query with unprocessed requests.
            request_limit: (int) Maximum number of requests to submit.
            properties: (list of str) Properties returned  in the dataframe.
                See the api link for a list of supported properties.
        """

        records = {}
        if not request_limit:  # interates through all results
            for i, entry in enumerate(query):
                records[i] = [getattr(entry, prop) for prop in properties]
        else:  # iterates through slice of results
            query[0]  # requests first page to reveal total results
            limit = min(query._N, query.k * request_limit)
            for i, entry in enumerate(query[0:limit]):
                records[i] = [getattr(entry, prop) for prop in properties]
        return DataFrame.from_dict(data=records, orient='index',
                                   columns=properties)


class RetrievalQuery(Query):
    """Provides additional methods for constructing class instances.
    """

    @classmethod
    def from_pymongo(cls, criteria, properties, request_size):
        """Generates an aflow Query object from pymongo-like arguments.

        Args:
            criteria: (dict) Pymongo-like query operator. See the
                AFLOWDataRetrieval.get_dataframe method for more details
            properties: (list of str) Properties returned  in the dataframe.
                See the api link for a list of supported properties.
            request_size: (int) Number of results to return per HTTP request.
        """
        # initializes query
        query = RetrievalQuery(batch_size=request_size)

        # adds filters to query
        query.add_pymongo_filters(criteria)

        # determines properties returned by query
        query.select(*[getattr(K, i) for i in properties])

        # supresses properties that may have been included in criteria
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
    ret = AFLOWDataRetrieval()
    item = ret.get_dataframe(criteria={'spacegroup_relax': {'$in': [216, 225]},
                                       'natoms': 3,
                                       'compound': 'Hf1Ni1Sn1'},
                             properties=['compound', 'enthalpy_formation_atom'],
                             request_size=100000, request_limit=12,
                             index_auid=True, autocast=False)
    print(type(item['enthalpy_formation_atom'][1]))
