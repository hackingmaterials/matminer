import re
import inspect
import sympy as sp
import matminer.models.mechanical_properties as mech
import yaml
import pandas as pd


with open('catalog.yaml', 'r') as f_catalog:
    catalog = yaml.load(f_catalog)


def decorate_dataframe(df):
    matched_props = []
    for col in df.columns:
        mcol = col.lower()
        mcol = re.sub(r'\(.*\)', '', mcol)
        mcol = re.sub(r'\'', '', mcol)
        for prop in catalog['properties']:
            if mcol.strip() in prop['spec']['name']:
                matched_props.append({'catalog_name': prop['spec']['name'].replace(' ', ''), 'df_name': col,
                                     'symbol': sp.symbols(prop['spec']['symbol']), 'units': prop['spec']['units']})

    mech_props = {c[0]: c[1] for c in inspect.getmembers(mech, inspect.isclass)}

    matched_mech_props_cls = []
    df_matchedcols = {}
    for m in matched_props:
        for mechprop in mech_props:
            if m['catalog_name'] in mechprop.lower():
                matched_mech_props_cls.append(mech_props[mechprop])
                df_matchedcols[m['symbol']] = m['df_name']

    eqns = []

    for matched_cls in matched_mech_props_cls:
        eqns.append(sp.Eq(matched_cls().equation()))

    for idx, row in df.iterrows():
        eqns_tosolve = eqns[:]
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
    # df.to_pickle('moduli.pkl')
    df1 = pd.read_pickle('moduli.pkl')
    df = df1.groupby(df1['material.chemicalFormula'], as_index=False).mean()
    print df
    new_df = decorate_dataframe(df)
    print new_df

