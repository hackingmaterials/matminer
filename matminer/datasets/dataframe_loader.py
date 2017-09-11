import pandas
import os


__author__ = "Kyle Bystrom <kylebystrom@berkeley.edu>, " \
             "Anubhav Jain <ajain@lbl.gov>"

module_dir = os.path.dirname(os.path.abspath(__file__))


def load_elastic_tensor():
    # ref: Jong, M. De, Chen, W., Angsten, T., Jain, A., Notestine, R., Gamst,
    # A., Sluiter, M., Ande, C. K., Zwaag, S. Van Der, Plata, J. J., Toher,
    # C., Curtarolo, S., Ceder, G., Persson, K. a & Asta, M. Charting the
    # complete elastic properties of inorganic crystalline compounds. Sci.
    # Data 2, 150009 (2015).
    return pandas.read_csv(os.path.join(module_dir, "elastic_tensor.csv"),
                           comment="#")


def load_piezoelectric_tensor():
    # ref: de Jong, M., Chen, W., Geerlings, H., Asta, M. & Persson, K. A.
    # A database to enable discovery and design of piezoelectric materials.
    # Sci. Data 2, 150053 (2015).
    return pandas.read_csv(os.path.join(module_dir, "piezoelectric_tensor.csv"), comment="#")


def load_dielectric_constant():
    # ref: Petousis, I., Mrdjenovich, D., Ballouz, E., Liu, M., Winston, D.,
    # Chen, W., Graf, T., Schladt, T. D., Persson, K. A. & Prinz, F. B.
    # High-throughput screening of inorganic compounds for the discovery of
    # novel dielectric and optical materials. Sci. Data 4, 160134 (2017).
    return pandas.read_csv(os.path.join(module_dir, "dielectric_constant.csv"),
                           comment="#")
