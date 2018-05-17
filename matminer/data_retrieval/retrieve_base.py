
__author__ = ["Alexander Dunn <ardunn@lbl.gov>",
             "Alireza Faghaninia <alireza.faghaninia@gmail.com>"]

class BaseDataRetrieval:
    """
    Abstract class to retrieve data from various material APIs while adhering to
    a quasi-standard format for querying.


    ## Implementing a new DataRetrieval class

    If you have an API which you'd like to incorporate into matminer's data
    retrieval tools, using BaseDataRetrieval is the preferred way of doing so.
    All DataRetrieval classes should subclass BaseDataRetrieval and implement
    the following:
        * get_dataframe()
        * api_link()

    Retrieving data should be done by the user with get_dataframe. Criteria
    should be a dictionary which will be used to form a query to the database.
    Properties should be a list which defines the columns that will be returned.
    While the 'criteria' and 'properties' arguments may have different valid
    values depending on the database, they should always have sensible formats
    and names if possible. For example, the user should be calling this:

    df = MyDataRetrieval().get_dataframe(criteria={'band_gap': 0.0},
                                         properties=['structure'])

    ...or this:

    df = MyDataRetrieval().get_dataframe(criteria={'band_gap': [0.0, 0.15]},
                                         properties=["density of states"])

    NOT this:

    df = MyDataRetrieval().get_dataframe(criteria={'query.bg[0] && band_gap': 0.0},
                                         properties=['Struct.page[Value]'])

    The implemented DataRetrieval class should handle the conversion from a
    'sensible' query to a query fit for the individual API and database.

    There may be cases where a 'sensible' query is not sufficient to define a
    query to the API; in this case, use the get_dataframe kwargs sparingly to
    augment the criteria, properties, or form of the underlying API query.

    A method for accessing raw DB data with an API-native query *may* be
    provided by overriding get_data. The link to the original API documentation
    *must* be provided by overriding api_link().

    ## Documenting a DataRetrieval class

    The class documentation for each DataRetrieval class must contain a brief
    description of the possible data that can be retrieved with the API source.
    It should also detail the form of the criteria and properties that can be
    retrieved with the class, and/or should link to a web page showing this
    information. The options of the class must all be defined in the `__init__`
    function of the class, and we recommend documenting them using the
    [Google style](https://google.github.io/styleguide/pyguide.html).
    """
    def api_link(self):
        """
        The link to comprehensive API documentation or data source.

        Returns:
            (str): A link to the API documentation for this DataRetrieval class.

        """
        raise NotImplementedError("api_link() is not defined!")

    def get_dataframe(self, criteria, properties, **kwargs):
        """
        Retrieve a dataframe of properties from the database which satisfy
        criteria.

        Args:
            criteria (dict): The name of each criterion is the key; the value
                or range of the criterion is the value.
            properties (list): Properties to return from the query matching
                the criteria. For example, ['structure', 'formula']

        Returns:
            (pandas DataFrame) The dataframe containing properties as columns
                and samples as rows.

        """
        raise NotImplementedError("get_dataframe() is not defined!")