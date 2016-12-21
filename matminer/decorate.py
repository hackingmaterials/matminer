import re
import inspect
import sympy as sp
from matminer.models import mechanical_properties
import yaml
import pandas as pd
from matminer.data_retrieval.retrieve_Citrine import CitrineDataRetrieval


with open('catalog.yaml', 'r') as f_catalog:
    catalog = yaml.load(f_catalog)


def decorate_dataframe(df):

    # Get properties matched from catalog
    catalog_matched_props = []

    for col in df.columns:
        mcol = col.lower()
        mcol = re.sub(r'\(.*\)', '', mcol)    # Remove parenthesis
        mcol = re.sub(r'\'', '', mcol)        # Remove apostrophes
        mcol = mcol.strip()

        for prop in catalog['properties']:
            if mcol in prop['spec']['name']:
                catalog_matched_props.append({'catalog_name': prop['spec']['name'].replace(' ', ''), 'df_name': col,
                                              'symbol': sp.symbols(prop['spec']['symbol']),
                                              'units': prop['spec']['units']})
            # TODO: check for alternate names

    # Get property class names and objects
    mech_props = {c[0].lower(): c[1] for c in inspect.getmembers(mechanical_properties, inspect.isclass)}

    prop_clsobjects = []    # Store class objects of matched property names
    df_matchedcols = {}     # Matched columns in df in 'catalog_symbol': 'col_name' format.
    for cmp in catalog_matched_props:
        if cmp['catalog_name'] in mech_props:
            prop_clsobjects.append(mech_props[cmp['catalog_name']])
            df_matchedcols[cmp['symbol']] = cmp['df_name']

    print mech_props
    print prop_clsobjects

    # Get equations of matched properties
    eqns = []
    for matched_cls in prop_clsobjects:
        eqns.append(sp.Eq(matched_cls().equation()))

    for idx, row in df.iterrows():

        eqns_tosolve = eqns[:]

        # add equation of symbol and its values from provided df
        for df_col in df_matchedcols:
            eqns_tosolve.append(sp.Eq(df_col, row[df_matchedcols[df_col]]))

        soln = sp.solve(eqns_tosolve)
        if soln:

            print idx, eqns_tosolve, soln
            df.loc[idx, "Calculated Poisson's ratio"] = round(soln[0][sp.S('nu')], 2)
    return df


if __name__ == '__main__':
    pd.set_option('display.width', 1000)
    # df1 = pd.read_pickle('39135_BMG.pkl')
    df = CitrineDataRetrieval().get_dataframe(data_set_id=150628, max_results=50)
    df = df.groupby(['chemicalFormula'], as_index=False).sum()
    print df
    new_df = decorate_dataframe(df)
    print new_df

