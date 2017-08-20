import pandas
import os

module_dir = os.path.dirname(os.path.abspath(__file__))


def load_elastic_tensor():
    return pandas.read_csv(os.path.join(module_dir, "ec.csv"))


def load_piezoelectric_tensor():
    return pandas.read_csv(os.path.join(module_dir, "piezo.csv"))


def load_dielectric_const_and_ref_ind():
    return pandas.read_csv(os.path.join(module_dir, "diel_ref.csv"))
