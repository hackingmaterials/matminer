from __future__ import division, unicode_literals, print_function

from pymatgen.core.composition import Composition

from matminer.descriptors.data import PymatgenData

__author__ = 'Saurabh Bajaj <sbajaj@lbl.gov>'


class AddDescriptor:
    """
    Code to add a descriptor column to a dataframe
    """

    def __init__(self, df, formula_colname='pretty_formula', separator='_'):
        """
        Args:
            df: dataframe to add the descriptor column to
            formula_colname: (str) name of the column containing the formula/composition
            separator: (str) separator to use in naming the new descriptor column

        Returns: None
        """
        self.df = df
        self.formula_colname = formula_colname
        self.separator = separator
        self.pmg_data = PymatgenData()

    def add_pmgdescriptor_column(self, descriptor, stat_function, stat_name):
        """
        Args:
            descriptor: (str) name of descriptor - must match the name in the source library
            stat_function: function to approximate the descriptor. For example, numpy.mean, numpy.std, etc.
            stat_name: (str) name of stat function to append to new descriptor column name

        Returns: dataframe with appended descriptor column
        """
        try:
            self.df[descriptor + self.separator + stat_name] = self.df[self.formula_colname]. \
                map(lambda x: stat_function(self.pmg_data.get_property(Composition(x), descriptor)))
        except ValueError:
            self.df.loc[descriptor + self.separator + stat_name] = None
        except AttributeError:
            raise ValueError('Invalid pymatgen Element attribute!')
        return self.df
