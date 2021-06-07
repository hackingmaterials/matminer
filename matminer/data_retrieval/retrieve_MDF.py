import pandas as pd
from matminer.data_retrieval.retrieve_base import BaseDataRetrieval

from mdf_forge.forge import Forge
from matminer.utils.flatten_dict import flatten_dict

__author__ = "Joseph Montoya <montoyjh@lbl.gov>"


class MDFDataRetrieval(BaseDataRetrieval):
    """
    MDFDataRetrieval is used to retrieve data from the Materials Data Facility
    database and convert them into a Pandas DataFrame. Note that invocation
    with full access to MDF will require authentication (see api_link) but an
    anonymous mode is supported, which can be used with anonymous=True as a
    keyword arg.

    Examples:
        >>>mdf_dr = MDFDataRetrieval(anonymous=True)
        >>>results = mdf_dr.get_dataframe({"elements":["Ag", "Be"], "source_names": ["oqmd"]})

        >>>results = mdf_dr.get_dataframe({"source_names": ["oqmd"],
        >>>          "match_ranges": {"oqmd.band_gap.value": [4.0, "*"]}})

    If you use this data retrieval class, please additionally cite:
    Blaiszik, B., Chard, K., Pruyne, J., Ananthakrishnan, R., Tuecke, S.,
    Foster, I., 2016. The Materials Data Facility: Data Services to Advance
    Materials Science Research. JOM 68, 2045–2052.
    https://doi.org/10.1007/s11837-016-2001-3
    """

    def __init__(self, anonymous=False, **kwargs):
        """
        Args:
            anonymous (bool): whether to use anonymous login (i. e. no
                globus authentication)
            **kwargs: kwargs for Forge, including index (globus search index
                to search on), local_ep, anonymous
        """

        self.forge = Forge(anonymous=anonymous, **kwargs)

    def api_link(self):
        return "https://github.com/materials-data-facility/forge"

    def get_dataframe(self, criteria, properties=None, unwind_arrays=True):
        """
        Retrieves data from the MDF API and formats it as a Pandas Dataframe

        Args:
            criteria (dict): options for keys are
                source_names ([str]): source names to include, e. g. ["oqmd"]
                elements ([str]): elements to include, e. g. ["Ag", "Si"]
                titles ([str]): titles to include, e. g. ["Coarsening of a
                    semisolid Al-Cu alloy"]
                tags ([str]): tags to include, e. g. ["outcar"]
                resource_types ([str]): resources to include, e. g. ["record"]
                match_fields ({}): field-value mappings to include, e. g.
                    {"oqmd.converged": True}
                exclude_fields ({}): field-value mappings to exclude, e. g.
                    {"oqmd.converged": False}
                match_ranges ({}): field-range mappings to include, e. g.
                    {"oqmd.band_gap.value": [1, 5]}, use "*" for no lower
                    or upper bound, e. g. {"oqdm.band_gap.value": [1, "*"]},
                exclude_ranges ({}): field-range mapping to exclude,
                    {"oqmd.band_gap.value": [3, "*"]} to exclude all
                    results with band gap higher than 3.
                raw (bool): whether or not to return raw (non-dataframe)
                    output, defaults to False
            unwind_arrays (bool): whether or not to unwind arrays in
                flattening docs for dataframe

        Returns (pandas.DataFrame):
            DataFrame corresponding to all documents from aggregated query
        """
        # Each of these fields has a "match_X" method in forge, do these first
        for key in ["source_names", "elements", "titles", "tags", "resource_types"]:
            if criteria.get(key):
                fn = getattr(self.forge, "match_{}".format(key))
                fn(criteria.get(key))

        # Each of these requires unpacking a dictionary and sometimes a range
        for key in ["match_fields", "exclude_fields", "match_ranges", "exclude_ranges"]:
            qvalue = criteria.get(key)
            if qvalue:
                fn = getattr(self.forge, key[:-1])  # remove 's' at end
                for field, value in qvalue.items():
                    if "ranges" in key:
                        fn(field, *value)
                    else:
                        fn(field, value)
        results = self.forge.aggregate()
        return make_dataframe(results, unwind_arrays=unwind_arrays)

    def get_data(self, squery, unwind_arrays=True, **kwargs):
        """
        Gets a dataframe from the MDF API from an explicit string
        query (rather than input args like get_dataframe).

        Args:
            squery (str): String for explicit query
            unwind_arrays (bool): whether or not to unwind arrays in
                flattening docs for dataframe
            **kwargs: kwargs for query

        Returns:
            dataframe corresponding to query

        """
        results = self.forge.aggregate(q=squery, **kwargs)
        return make_dataframe(results, unwind_arrays=unwind_arrays)

    def citations(self):
        return [
            "@article{Blaiszik2019,"
            "doi = {10.1557/mrc.2019.118},"
            "url = {https://doi.org/10.1557/mrc.2019.118},"
            "year = {2019},"
            "month = sep,"
            "publisher = {Springer Science and Business Media {LLC}},"
            "volume = {9},"
            "number = {4},"
            "pages = {1125--1133},"
            "author = {Ben Blaiszik and Logan Ward and Marcus Schwarting and Jonathon Gaff "
            "and Ryan Chard and Daniel Pike and Kyle Chard and Ian Foster},"
            "title = {A data ecosystem to support machine learning in materials science},"
            "journal = {{MRS} Communications}"
            "}",
            "@article{Blaiszik2016,"
            "doi = {10.1007/s11837-016-2001-3},"
            "url = {https://doi.org/10.1007/s11837-016-2001-3},"
            "year = {2016},"
            "month = jul,"
            "publisher = {Springer Science and Business Media {LLC}},"
            "volume = {68},"
            "number = {8},"
            "pages = {2045--2052},"
            "author = {B. Blaiszik and K. Chard and J. Pruyne and R. Ananthakrishnan and "
            "S. Tuecke and I. Foster},"
            "title = {The Materials Data Facility: Data Services to Advance Materials "
            "Science Research},"
            "journal = {{JOM}}"
            "}",
        ]


# TODO: could add parallel functionality, but doesn't seem to be too slow
# TODO: also might be useful to handle units more intelligently
def make_dataframe(docs, unwind_arrays=True):
    """
    Formats raw docs returned from MDF API search into a dataframe

    Args:
        docs [{}]: list of documents from forge search
            or aggregation

    Returns: DataFrame corresponding to formatted docs

    """
    flattened_docs = [flatten_dict(doc, unwind_arrays=unwind_arrays) for doc in docs]
    df = pd.DataFrame(flattened_docs)
    return df
