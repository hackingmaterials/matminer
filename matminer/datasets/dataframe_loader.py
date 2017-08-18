import pandas

def load_df(filename):
	return pandas.read_csv(filename)

def load_elastic_tensor():
	return load_df("ec.csv")

def load_piezoelectric_tensor():
	return load_df("piezo.csv")

def load_dielectric_const_and_ref_ind():
	return load_df("diel_ref")
