"""
Functions used for auto-generating featurizer tables.
"""
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
    for sc in scnames:
        scdict = {"name": sc.__name__}
        scdict["doc"] = sc.__doc__.splitlines()[1].lstrip()
        scdict["module"] = sc.__module__
        scdict["type"] = sc.__module__.split(".")[-1]
        subclasses.append(scdict)

    df = pd.DataFrame(subclasses)
    print(mmfeat)
    print(mmdes)

    for ftype in np.unique(df['type']):
        dftable = df[df['type'] == ftype]
        dftable['codename'] = [":code:`" + n + "`" for n in dftable['name']]

        ftype_border = "-" * len(ftype)
        mod = "(" + dftable['module'].iloc[0] + ")"
        des_border = "-" * len(mod_summs[ftype])

        print(ftype_border)
        print(ftype)
        print(ftype_border)
        print(mod_summs[ftype])
        print(des_border + "\n")
        print(mod)

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
