import pandas as pd

from mdf_forge.forge import Forge
from matminer.utils.flatten_dict import flatten_dict

__author__ = 'Joseph Montoya <montoyjh@lbl.gov>'


class MDFDataRetrieval:
    """
    MDFDataRetrieval is used to retrieve data from the
    Materials Data Facility database and convert them
    into a Pandas dataframe.  Note that invocation with
    full access to MDF will require authentication via
    https://materialsdatafacility.org/, but an anonymous
    mode is supported, which can be used with
    anonymous=True as a keyword arg.

    Examples:
        >>>mdf_dr = MDFDataRetrieval(anonymous=True)
        >>>results = mdf_dr.search(elements=["Ag", "Be"], sources=["oqmd"])

        >>>results = mdf_dr.search(sources=['oqmd'],
        >>>               match_ranges={"oqmd.band_gap.value": [4.0, "*"]})
    """

    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: kwargs for Forge, including index (globus search index
                to search on), local_ep, anonymous
        """

        self.forge = Forge(**kwargs)

    def search(self, sources=None, elements=None, titles=None, tags=None,
               resource_types=None, match_fields=None, exclude_fields=None,
               match_ranges=None, exclude_ranges=None, raw=False,
               unwind_arrays=True):
        """

        Args:
            sources ([str]): source names to include, e. g. ["oqmd"]
            elements ([str]): elements to include, e. g. ["Ag", "Si"]
            titles ([str]): titles to include, e. g. ["Coarsening of a semisolid
                Al-Cu alloy"]
            tags ([str]): tags to include, e. g. ["outcar"]
            resource_types ([str]): resources to include, e. g. ["record"]
            match_fields ({}): field-value mappings to include, e. g.
                {"oqdm.converged": True}
            exclude_fields ({}): field-value mappings to exclude, e. g.
                {"oqdm.converged": False}
            match_ranges ({}): field-range mappings to include, e. g.
                {"oqdm.band_gap.value": [1, 5]}, use "*" for no lower
                or upper bound, e. g. {"oqdm.band_gap.value": [1, "*"]},
            exclude_ranges ({}): field-range mapping to exclude,
                {"oqdm.band_gap.value": [3, "*"]} to exclude all
                results with band gap higher than 3.
            raw (bool): whether or not to return raw (non-dataframe)
                output, defaults to False
            unwind_arrays (bool): whether or not to unwind arrays in
                flattening docs for dataframe

        Returns:
            DataFrame corresponding to all documents from aggregated query
        """

        # self.forge.reset_query()

        search_args = locals()

        # Each of these fields has a "match_X" method in forge, do these first
        for query_field in ["sources", "elements", "titles", "tags",
                            "resource_types"]:
            if search_args.get(query_field):
                fn = getattr(self.forge, "match_{}".format(query_field))
                fn(search_args.get(query_field))

        # Each of these requires unpacking a dictionary and sometimes a range
        for query_field in ["match_fields", "exclude_fields", "match_ranges",
                            "exclude_ranges"]:
            query_value = search_args.get(query_field)
            if query_value:
                fn = getattr(self.forge, query_field[:-1])  # remove 's' at end
                for field, value in query_value.items():
                    if "ranges" in query_field:
                        fn(field, *value)
                    else:
                        fn(field, value)

        results = self.forge.aggregate()

        # Make into DataFrame
        if raw:
            return results
        else:
            return make_dataframe(results, unwind_arrays=unwind_arrays)


    def search_by_query(self, query, unwind_arrays=True,
                        raw=False, **kwargs):
        """

        Args:
            query (str): String for explicit query
            raw (bool): whether or not to return raw (non-dataframe)
                output, defaults to False
            unwind_arrays (bool): whether or not to unwind arrays in
                flattening docs for dataframe
            **kwargs: kwargs for query

        Returns:
            aggregated data corresponding to query

        """
        results = self.forge.aggregate(q=query, **kwargs)
        if raw:
            return results
        else:
            return make_dataframe(results, unwind_arrays=unwind_arrays)



#TODO: could add parallel functionality, but doesn't seem to be too slow
#TODO: also might be useful to handle units more intelligently
def make_dataframe(docs, unwind_arrays=True):
    """
    Formats raw docs returned from search into a dataframe

    Args:
        docs [{}]: list of documents from forge search
            or aggregation

    Returns: DataFrame corresponding to formatted docs

    """
    flattened_docs = [flatten_dict(doc, unwind_arrays=unwind_arrays)
                      for doc in docs]
    df = pd.DataFrame(flattened_docs)
    return df





