import pandas as pd
from matminer.descriptors.composition_features import *

__author__ = 'Saurabh Bajaj <sbajaj@lbl.gov>'


# TODO: Check how to automatically get stats (mean, median,..) from the descriptor column and use them to set limits
# for plot colors
# TODO: Check how to set legends in plots (return them here and pass them onto plot_xy()


class AddDescriptor:
    def __init__(self, df=None, formula_colname="reduced_cell_formula", separator=":"):
        self.df = df
        self.formula_colname=formula_colname
        self.separator=separator

    def add_pmgdescriptor_column(self, descriptor, stat):
        # TODO: requires lots of code cleanup
        # TODO: this is likely super inefficient

        if stat == 'mean':
            stat_function = get_mean
        elif stat == 'std':
            stat_function = get_std
        elif stat == 'maxmin':
            stat_function = get_maxmin
        else:
            raise ValueError('Invalid stat name. Must be one of "mean", "std", and "maxmin"')
        for i, row in self.df.iterrows():
            try:
                self.df.loc[i, descriptor + self.separator + stat] = stat_function(
                    get_pymatgen_descriptor(row[self.formula_colname], descriptor))
            except ValueError:
                self.df.loc[i, descriptor + self.separator + stat] = None
            except AttributeError as e:
                print(e)
                print('Invalid pymatgen Element attribute!')
        return self.df

if __name__ == '__main__':
    print(AddDescriptor())
