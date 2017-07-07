from __future__ import division, unicode_literals, print_function

from functools import reduce

from pymatgen import Composition, MPRester, Element

from matminer.descriptors.data import cohesive_energy_data, PymatgenData


def get_cohesive_energy(comp):
    """
    Get cohesive energy of compound by subtracting elemental cohesive energies from the formation
    energy of the compund. Elemental cohesive energies are taken from
    http://www.knowledgedoor.com/2/elements_handbook/cohesive_energy.html.
    Most of them are taken from "Charles Kittel: Introduction to Solid State Physics, 8th edition.
    Hoboken, NJ: John Wiley & Sons, Inc, 2005, p. 50."

    Args:
        comp: (str) compound composition, eg: "NaCl"

    Returns: (float) cohesive energy of compound

    """
    el_amt_dict = Composition(comp).get_el_amt_dict()

    # Get formation energy of most stable structure from MP
    struct_lst = MPRester().get_data(comp)
    if len(struct_lst) > 0:
        struct_lst = sorted(struct_lst, key=lambda e: e['energy_per_atom'])
        most_stable_entry = struct_lst[0]
        formation_energy = most_stable_entry['formation_energy_per_atom']
    else:
        raise ValueError('No structure found in MP for {}'.format(comp))

    # Subtract elemental cohesive energies from formation energy
    cohesive_energy = formation_energy
    for el in el_amt_dict:
        cohesive_energy -= el_amt_dict[el] * cohesive_energy_data[el]

    return cohesive_energy


def band_center(comp):
    """
    Estimate absolution position of band center using geometric mean of electronegativity
    Ref: Butler, M. a. & Ginley, D. S. Prediction of Flatband Potentials at
    Semiconductor-Electrolyte Interfaces from Atomic Electronegativities.
    J. Electrochem. Soc. 125, 228 (1978).

    Args:
        comp: (Composition)

    Returns: (float) band center

    """
    prod = 1.0
    for el, amt in comp.get_el_amt_dict().iteritems():
        prod = prod * (Element(el).X ** amt)

    return -prod ** (1 / sum(comp.get_el_amt_dict().values()))


def get_holder_mean(data_lst, power):
    """
    Get Holder mean

    Args:
        data_lst: (list/array) of values
        power: (int/float) non-zero real number

    Returns: Holder mean

    """
    # Function for calculating Geometric mean
    geomean = lambda n: reduce(lambda x, y: x * y, n) ** (1.0 / len(n))

    # If power=0, return geometric mean
    if power == 0:
        return geomean(data_lst)

    else:
        total = 0.0
        for value in data_lst:
            total += value ** power
        return (total / len(data_lst)) ** (1 / float(power))


def add_element_property_column(df, property_name, filter_func, formula_column="formula",
                                data_source=None):
    """
    Add elemental property data filtered by the 'filter_func' for each formula in the given
    dataframe's 'formula_column' to the given dataframe.

    Args:
        df (pandas.DataFrame): the dataframe containing the list of formulae to process.
        property_name (str): elemental property name. Note: must be supported by the given
            data_source.
        filter_func (func): the function to apply to the raw elemental property values
        formula_column (str): the name of the column in the dataframe containing the formulae
            to process
        data_source (data.AbstractData): data object that will be used to obtain the elemental
            properties. eg. MagpieData() or PymatgenData()

    Returns:
        pandas.DataFrame: modified dataframe with the added elemental property column.
    """
    property_column_name = "{}_{}".format(property_name, filter_func.__name__)
    data_source = data_source or PymatgenData()
    try:
        map_func = lambda x: filter_func(data_source.get_property(Composition(x), property_name))
        df[property_column_name] = df[formula_column].map(map_func)
    except ValueError:
        df.loc[property_column_name] = None
    except AttributeError:
        raise ValueError('Invalid Elemental property!')
    return df
