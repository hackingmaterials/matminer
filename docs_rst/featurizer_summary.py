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
from matminer.featurizers.base import BaseFeaturizer

__authors__ = 'Alex Dunn <ardunn@lbl.gov>'

# Update the following dictionary if any new modules are added. Each key is
# the name of the module and each value is a high level summary of its contents.
mod_summs = {"structure": "Generating features based on a material's crystal structure.\n",
             "site": "Features from individual sites in a material's crystal structure.\n",
             "dos": "Features based on a material's electronic density of states.\n",
             "base": "Parent classes and meta-featurizers.\n",
             "composition": "Features based on a material's composition.\n",
             "function": "Classes for expanding sets of features calculated with other featurizers.\n",
             "bandstructure": "Features derived from a material's electronic bandstructure.\n"}

url_base = " `[more] <https://hackingmaterials.github.io/matminer/matminer.featurizers.html#"

def generate_tables():
    """
    Generate nicely formatted tables of all features in RST format.

    Args:
        None

    Returns:
        tables ([str]): A list of formatted strings, where each entry is a
            separate table representing one module.
    """

    mmfeat = "===========\nFeaturizers\n===========\n"
    mmdes = "Below, you will find a description of each featurizer, listed in " \
            "tables grouped by module.\n"
    tables = [mmfeat, mmdes]
    subclasses = []
    for sc in BaseFeaturizer.__subclasses__() + [BaseFeaturizer]:
        scdict = {"name": sc.__name__}
        scdict["doc"] = sc.__doc__.splitlines()[1].lstrip()
        scdict["module"] = sc.__module__
        scdict["type"] = sc.__module__.split(".")[-1]
        subclasses.append(scdict)

    df = pd.DataFrame(subclasses)

    for ftype in np.unique(df['type']):
        dftable = df[df['type'] == ftype]
        dftable['codename'] = [":code:`" + n + "`" for n in dftable['name']]
        mod = "\n(" + dftable['module'].iloc[0] + ")\n\n"
        namelen = max([len(n) for n in dftable['codename']])
        # doclen = max([len(d) for d in dftable['doc']])
        doclen = 400
        borderstr = "=" * namelen + "   " + "=" * doclen + "\n"
        headerstr = "Name" + " " * (namelen - 1) + "Description\n"
        tablestr = ""
        for i, n in enumerate(dftable['codename']):
            url = url_base + dftable["module"].iloc[0] + "." + \
                  dftable["name"].iloc[i] + ">`_"
            tablestr += n + " " * (namelen - len(n) + 3) + \
                        dftable['doc'].iloc[i] + url + "\n"


        ftype_border = "\n" + "-" * len(ftype) + "\n"
        des_border = "-" * len(mod_summs[ftype]) + "\n"
        tables.append(ftype_border + ftype + ftype_border + mod_summs[ftype] +
                      des_border + mod + borderstr + headerstr + borderstr +
                      tablestr + borderstr + "\n\n")

    return tables

if __name__ == "__main__":
    for t in generate_tables():
        print(t)