import pandas as pd

from mdf_forge.forge import Forge

__author__ = 'Joseph Montoya <montoyjh@lbl.gov>'


class MDFDataRetrieval:
    """
    MDFDataRetrieval is used to retrieve data from the Materials Data Facility 
    database, print the results, and convert them into an indexed Pandas
    dataframe.
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
               match_ranges=None, exclude_ranges=None, reset=True):
        """
        
        Args:
            sources ([str]): source names to include
            elements ([str]): elements to include
            titles ([str]): titles to include 
            tags ([str]): tags to include 
            resource_types ([str]): resources to include 
            match_fields ({}): field-value mappings to include
            exclude_fields ({}): field-value mappings to exclude
            match_ranges ({}): field-range mappings to include 
            exclude_ranges ({}): field-range mapping to exclude 
            
        Returns:
            list of documents corresponded to the aggregated query
        """

        self.forge.reset_query()

        search_args = locals()

        # Each of these fields has a "match_X" method in forge, do these first
        for query_field in ["sources", "elements", "titles", "tags",
                            "resource_types"]:
            if locals.get(query_field):
                fn = getattr(self.forge, "match_{}".format(query_field))
                fn(locals.get(query_field))

        # Each of these requires unpacking a dictionary and sometimes a range
        for query_field in ["match_fields", "exclude_fields", "match_ranges",
                            "exclude_ranges"]:
            query_value = locals.get(query_field)
            if query_value:
                fn = getattr(self.forge, query_field[:-1])
                for field, value in query_value.items():
                    if "ranges" in query_field:
                        fn(field, *value)
                    else:
                        fn(field, value)

        return self.forge.aggregate()


    def search_by_query(self, query, **kwargs):
        """
        
        Args:
            query (str): String for explicit query 
            **kwargs: kwargs for query

        Returns:
            aggregated data corresponding to query

        """
        return self.forge.aggregate(q=query, **kwargs)

    def get_dataframe(self, *args):
        pass
