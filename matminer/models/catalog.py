import pandas as pd

__author__ = 'Saurabh Bajaj <sbajaj@lbl.gov>'

#TODO: add altername symbol and property names
#TODO: take care of units

if __name__ == "__main__":
    df_catalog = pd.DataFrame(columns=['symbol', 'property_name', 'si_units'])
    properties = []
    properties.append(["N", "number of atoms", ""])
    properties.append(["m", "mass", "kg"])
    properties.append(["V", "volume", "m^3"])
    properties.append(["rho", "density", "kg/m^3"])
    properties.append(["nu", "Poisson's ratio", ""])
    properties.append(["E", "Young's modulus", "N/m^2"])
    properties.append(["K", "Bulk modulus", "N/m^2"])
    properties.append(["G", "Shear modulus", "N/m^2"])
    for idx, prop in enumerate(properties):
        df_catalog.loc[idx] = prop
    df_catalog.to_pickle('df_catalog.pkl')
