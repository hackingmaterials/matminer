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

    def generate_match(self, fields, values, start=None):
        """
        
        Args:
            fields ([str]): match fields
            values ([str]): match values
            start (match): starting match object

        Returns:
            match object

        """
        if start is None:
            match = self.forge
        for field, value in zip(fields, values):
            match.match_field(field, value)
        return match

    def search(self, sources, filter=None,
               keep_mdf_metadata=False):
        forge = self.forge.match_sources(sources)

        return match.search()

    def search_by_query(self, query, **kwargs):

    def get_dataframe(self, *args):
        pass
