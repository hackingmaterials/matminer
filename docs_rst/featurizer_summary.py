"""
Functions used for auto-generating featurizer tables.
"""
import importlib

import numpy as np
import pandas as pd
from matminer.featurizers import base
from matminer.featurizers import composition
from matminer.featurizers import site
from matminer.featurizers import structure
from matminer.featurizers import dos
from matminer.featurizers import bandstructure
from matminer.featurizers import function
from matminer.featurizers import conversions
from matminer.featurizers.base import BaseFeaturizer

__authors__ = 'Alex Dunn <ardunn@lbl.gov>'

# Update the following dictionary if any new modules are added. Each key is
# the name of the module and each value is a high level summary of its contents.
mod_summs = {
    "structure": "Generating features based on a material's crystal structure.",
    "site": "Features from individual sites in a material's crystal structure.",
    "dos": "Features based on a material's electronic density of states.",
    "base": "Parent classes and meta-featurizers.",
    "composition": "Features based on a material's composition.",
    "function": "Classes for expanding sets of features calculated with other featurizers.",
    "bandstructure": "Features derived from a material's electronic bandstructure.",
    "conversions": "Conversion utilities."}

url_base = " `[more] <https://hackingmaterials.github.io/matminer/matminer.featurizers.html#"

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def generate_tables():
    """
    Generate nicely formatted tables of all features in RST format.

    Args:
        None

    Returns:
        Prints a formatted string, where each main entry is a separate table
        representing one module of featurizers.
    """

    mmfeat = "====================\nTable of Featurizers\n====================\n"
    mmdes = "Below, you will find a description of each featurizer, listed in " \
            "tables grouped by module.\n"
    subclasses = []
    scnames = BaseFeaturizer.__subclasses__() + [BaseFeaturizer]
    scnames += conversions.ConversionFeaturizer.__subclasses__()

    allowable_types = ["base", "composition", "site", "structure", "bandstructure", "dos", "function", "conversions"]
    for sc in scnames:
        scdict = {"name": sc.__name__}
        scdict["doc"] = sc.__doc__.splitlines()[1].lstrip()
        scdict["module"] = sc.__module__

        module_tree = sc.__module__.split(".")
        for t in allowable_types:
            if t in module_tree:
                scdict["type"] = t
                break
        else:
            raise ValueError(f"Module {module_tree} does not contain any allowable module types!")


        if len(module_tree) == 4:
            scdict["subtype"] = ".".join((module_tree[-2], module_tree[-1]))
            m = importlib.import_module(scdict["module"])
            if m.__doc__:
                scdict["subdoc"] = m.__doc__.replace("\n", " ")
            else:
                raise ValueError("no doc for submodule ", scdict["module"])
        else:
            scdict["subtype"] = None
            scdict["subdoc"] = None

        subclasses.append(scdict)


    df = pd.DataFrame(subclasses)
    # print(mmfeat)
    print(mmdes)


    for ftype in np.unique(df['type']):
        dftable = df[df['type'] == ftype]
        if not dftable["subtype"].isna().any():
            for i, subtype in enumerate(np.unique(dftable["subtype"])):
                dfsubtable = dftable[df["subtype"] == subtype]
                if i == 0:
                    big_header = ftype
                else:
                    big_header = None

                generate_table(dfsubtable, big_header=big_header, little_header=subtype)
        else:
            generate_table(dftable, big_header=ftype)



def generate_table(dftable, big_header=None, little_header=None):
    dftable['codename'] = [":code:`" + n + "`" for n in dftable['name']]

    mod = ":code:`" + dftable['module'].iloc[0] + "`"

    if big_header:
        ftype_border = "-" * len(big_header)
        des_border = "-" * len(mod_summs[big_header])
        print(ftype_border)
        print(big_header)
        print(ftype_border)
        print(mod_summs[big_header])
        print(des_border + "\n")

        if not little_header:
            print(mod)

    if little_header:

        printable_little_header = little_header.split(".")[-1]
        fsubtype_border = "_" * len(printable_little_header)

        print(printable_little_header)
        print(fsubtype_border)
        print(mod +"\n\n")
        print(dftable["subdoc"].iloc[0])

    print("\n.. list-table::")
    print("   :align: left")
    print("   :widths: 30 70")
    # print("   :width: 70%")
    print("   :header-rows: 1\n")
    print("   * - Name")
    print("     - Description")
    for i, n in enumerate(dftable['codename']):
        # url = url_base + dftable["module"].iloc[0] + "." + \
        #       dftable["name"].iloc[i] + ">`_"
        url = ""

        print(f"   * - {n}")
        description = dftable["doc"].iloc[i]
        print(f"     - {description} {url}    ")
    print("\n\n")


if __name__ == "__main__":
    generate_tables()
