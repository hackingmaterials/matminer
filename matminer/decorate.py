import re
import inspect
import sympy as sp
import matminer.models.mechanical_properties as mech
import yaml
import pandas as pd


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

    # Get property class names and objects
    mech_props = {c[0]: c[1] for c in inspect.getmembers(mech, inspect.isclass)}

    print catalog_matched_props
    print mech_props

    prop_clsobjects = []    # Store class objects of matched property names
    df_matchedcols = {}     # Matched columns in df in 'catalog_symbol': 'col_name' format.
    for cmp in catalog_matched_props:
        for cls in mech_props:
            if cmp['catalog_name'] in cls.lower():
                prop_clsobjects.append(mech_props[cls])
                df_matchedcols[cmp['symbol']] = cmp['df_name']

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
    # df = CitrineDataRetrieval().get_dataframe(data_set_id=39135)
    # df.to_pickle('39135_BMG.pkl')
    # print pd.read_pickle('bulkmoduli.pkl')
    df1 = pd.read_pickle('39135_BMG.pkl')
    df = df1.groupby(df1['material.chemicalFormula'], as_index=False).mean()
    print df
    new_df = decorate_dataframe(df)
    print new_df

