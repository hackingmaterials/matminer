from functools import reduce

from pandas import DataFrame

from pymatgen.core.structure import Structure

from matminer.data_retrieval.retrieve_base import BaseDataRetrieval

from aflow import K  # module of aflow Keyword properties
from aflow.caster import cast
from aflow.control import Query
from aflow.entries import AflowFile

__author__ = ["Maxwell Dylla <280mtd@gmail.com>"]


class AFLOWDataRetrieval(BaseDataRetrieval):
    """Retrieves data from the AFLOW database.

    AFLOW uses the AFLUX API syntax, and the aflow library handles the HTTP
    requests for material properties. Note that this helper library is not an
    official repository of the AFLOW consortium. However, this library does
    dynamically generate the keywords supported by the AFLUX API from their
    servers, which makes it robust against changes in the AFLOW system.

    If you use this data retrieval class, please additionally cite:
    Rose, F., Toher, C., Gossett, E., Oses, C., Nardelli, M.B., Fornari, M.,
    Curtarolo, S., 2017. AFLUX: The LUX materials search API for the AFLOW
    data repositories. Computational Materials Science 137, 362–370.
    https://doi.org/10.1016/j.commatsci.2017.04.036

    """

    def api_link(self):
        return "https://rosenbrockc.github.io/aflow/index.html"

    def get_dataframe(
        self,
        criteria,
        properties,
        files=None,
        request_size=10000,
        request_limit=0,
        index_auid=True,
    ):
        """Retrieves data from AFLOW in a DataFrame format.

        The method builds an AFLUX API query from pymongo-like filter criteria
        and requested properties. Then, results are collected over HTTP. Note
        that the "compound", "auid", and "aurl" fields are always returned.

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
            files: (list of str) For convienience, specific files may also be
                downloaded as pymatgen objects. Each file download is collected
                by a seperate HTTP request (read slow). The default behavior is
                to return none of these objects. Supported files:
                    "prototype_structure" - the prototype structure
                    "input_structure" - the input structure
                    "band_structure" - TODO
                    "dos" - TODO
            request_size: (int) Number of results to return per HTTP request.
            request_limit: (int) Maximum number of requests to submit. The
                default behavior is to request all matching records.
            index_auid: (bool) Whether to set the "AFLOW unique identifier" as
                the index of the DataFrame.

        Returns (pandas.DataFrame): The data requested from the AFLOW database.
        """

        # ensures that 'auid' is in requested properties if desired for index
        if index_auid and ("auid" not in properties):
            properties.append("auid")

        # generates a query for submitting HTTP requests to AFLOW servers
        query = RetrievalQuery.from_pymongo(criteria, properties, request_size)

        # submits HTTP requests and collects results
        df = self._collect_requests(query, request_limit)

        # casts each column into the correct data-type
        for keyword in df.columns.values:
            df[keyword] = self._cast_series(df[keyword])

        # collects the relaxed structures if requested
        if "structure" in files:
            df["structure"] = [self.get_relaxed_structure(url) for url in df["aurl"].values]

        # sets the auid as the index if desired
        if index_auid:
            df.set_index("auid", inplace=True)

        return df

    def citations(self):
        return [
            "@article{Curtarolo2012,"
            "doi = {10.1016/j.commatsci.2012.02.005},"
            "url = {https://doi.org/10.1016/j.commatsci.2012.02.005},"
            "year = {2012},"
            "month = jun,"
            "publisher = {Elsevier {BV}},"
            "volume = {58},"
            "pages = {218--226},"
            "author = {Stefano Curtarolo and Wahyu Setyawan and Gus L.W. Hart and Michal "
            "Jahnatek and Roman V. Chepulskii and Richard H. Taylor and Shidong Wang and "
            "Junkai Xue and Kesong Yang and Ohad Levy and Michael J. Mehl and Harold T. "
            "Stokes and Denis O. Demchenko and Dane Morgan},"
            "title = {{AFLOW}: An automatic framework for high-throughput materials discovery},"
            "journal = {Computational Materials Science}"
            "}"
        ]

    @staticmethod
    def get_relaxed_structure(aurl):
        """Collects the relaxed structure as a pymatgen.Structure.

        Args:
            aurl: (str) The url for the material entry in AFLOW.

        Returns: (pymatgen.Structure) The relaxed structure.
        """

        # downloads the file as a string
        file = AflowFile(aurl, "CONTCAR.relax.vasp")()  # calling induces dwnld

        # returns the python object
        return Structure.from_str(file, fmt="poscar")

    @staticmethod
    def _cast_series(series):
        """Casts AFLOW data (pandas Series) as the appropriate python type.

        Args:
            series: (pandas.Series) Str data to cast. The name attribute should
                correspond to a string representation of an aflow.Keyword

        Returns: (list) Data casted as the appropriate python object.
        """

        aflow_type, keyword = getattr(K, series.name).atype, series.name
        return [cast(aflow_type, keyword, i) for i in series.values]

    @staticmethod
    def _collect_requests(query, request_limit):
        """Collects the string-casted results of a query.

        Args:
            query: (aflow.control.Query) A query with unprocessed requests.
            request_limit: (int) Maximum number of requests to submit.

        Returns: (DataFrame) Results collected from the query.
        """

        # requests the first page of results to determine number of pages
        query._request(1, query.k)
        page_limit = (query._N // query.k) + 1
        if request_limit and (page_limit > request_limit):
            page_limit = request_limit

        # requests the remaining pages
        for page in range(2, page_limit + 1):
            query._request(page, query.k)

        # collects request responses
        records = {}
        for page in range(1, page_limit + 1):
            records.update(query.responses[page])
        return DataFrame.from_dict(data=records, orient="index")


class RetrievalQuery(Query):
    """Provides instance constructors for pymongo-like queries."""

    @classmethod
    def from_pymongo(cls, criteria, properties, request_size):
        """Generates an aflow Query object from pymongo-like arguments.

        Args:
            criteria: (dict) Pymongo-like query operator. See the
                AFLOWDataRetrieval.get_DataFrame method for more details
            properties: (list of str) Properties returned in the DataFrame.
                See the api link for a list of supported properties.
            request_size: (int) Number of results to return per HTTP request.
                Note that this is similar to "limit" in pymongo.find.
        """
        # initializes query
        query = cls(batch_size=request_size)

        # adds filters to query
        query._add_filters(criteria)

        # determines properties returned by query
        query.select(*[getattr(K, i) for i in properties])

        # suppresses properties that may have been included as search criteria
        # but are not requested properties, which the user wants returned
        excluded_keywords = set(criteria.keys()) - set(properties)
        query.exclude(*[getattr(K, i) for i in excluded_keywords])

        return query

    def _add_filters(self, pymongo_query):
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

                    if inner_key == "$in":
                        self.filter(
                            reduce(
                                lambda x, y: (x | y),
                                map(lambda z: (keyword == z), inner_value),
                            )
                        )

                    elif inner_key == "$gt":
                        self.filter((keyword > inner_value))

                    elif inner_key == "$lt":
                        self.filter((keyword < inner_value))

                    elif inner_key == "$not":
                        self.filter((~(keyword == inner_value)))

                    else:
                        raise Exception("Only $in, $gt, $lt, and $not are supported!")

            else:  # handles simple equivalence
                self.filter(keyword == value)
