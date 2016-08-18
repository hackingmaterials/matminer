import pandas as pd
from matminer.descriptors.composition_features import *

__author__ = 'Saurabh Bajaj <sbajaj@lbl.gov>'


# TODO: Check how to automatically get stats (mean, median,..) from the descriptor column and use them to set limits
# for plot colors
# TODO: Check how to set legends in plots (return them here and pass them onto plot_xy()


class AddDescriptor:
    def __init__(self, df=None, df_name=None):
        if df == df_name and df is None:
            raise ValueError('Atleast one of dataframe or pickled dataframe object are required')
        elif df is not None and df_name is not None:
            raise ValueError('Only one argument, dataframe or a pickled dataframe object, can be passed')
        elif df is not None:
            self.df = df
        elif df_name is not None:
            self.df = pd.read_pickle(df_name)

    def add_pmgdescriptor_column(self, descriptor, stat):
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
                self.df.loc[i, descriptor + '_' + stat] = stat_function(
                    get_element_data(row['reduced_cell_formula'], descriptor))
            except ValueError:
                self.df.loc[i, descriptor + '_' + stat] = None
            except AttributeError as e:
                print(e)
                print('Invalid pymatgen Element attribute!')
        return self.df


if __name__ == '__main__':
    print(AddDescriptor())
