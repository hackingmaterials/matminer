import pandas
import os

LOAD_PATH = os.path.dirname(os.path.abspath(__file__))

def load_df(filename):
	return pandas.read_csv(filename)

def load_elastic_tensor():
	return load_df(LOAD_PATH+"/ec.csv")

def load_piezoelectric_tensor():
	return load_df(LOAD_PATH+"/piezo.csv")

def load_dielectric_const_and_ref_ind():
	return load_df(LOAD_PATH+"/diel_ref.csv")
