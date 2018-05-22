import pandas as pd


class BaseGenerator:
    """Base class for tools that generate materials

    ## Using a Data Generator

    The purpose of the generator classes are to build a search space of materials.

    The initializer for each class contains the options for the search space (e.g., the names of
    the elements used to generate compositions.

    The simplest mechanism for generating entries is to call :code:`generate_entries`, which
    returns a generator. You can use generators in for loops, but may find it useful to first
    convert the generator list (e.g., :code:`list(gen.generate_entries())`).

    You can also directly generate a DataFrame by calling :code:`generate_dataframe`, which takes
    the desired column name for the entries.

    ## Implementing a Data Generator

    The only operations that need be implemented are :code:`generate_entries` and
    :code:`__init__`. :code:`generate_entries` simply generates (i.e., using :code:`yield`)
    materials in the desired format. :code:`__init__` takes any options for the feature generators

    ## Documenting a Data Generator

    Following the procedure for the featurizers, generators that produce different kinds of
    materials data must be stored in different modules. The only required documentation is for
    the initializer, which takes all of the operations of the class."""

    def generate_entries(self):
        """Generate entries

        Returns:
            Generator that produces a series of entries
        """
        raise NotImplementedError

    def generate_dataframe(self, column_name):
        """Generate entries an dstore them in a DataFrame

        Args:
            column_name (str): Name of the column in which to store the entries
        Returns:
            (DataFrame) entries stored in one column of a DataFrame
        """
        return pd.DataFrame({column_name: list(self.generate_entries())})
