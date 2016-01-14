from citrination_client import CitrinationClient
import os
import time
import pandas as pd
from tqdm import tqdm
from pandas.io.json import json_normalize
from pymatgen import Composition, Element


class CitrineDataRetrieval:
    def __init__(self, api_key=None, term=None, formula=None, property=None, contributor=None, reference=None,
                 min_measurement=None,
                 max_measurement=None, from_record=None, per_page=None, data_set_id=None):
        # TODO: create/format docstrings for this and all other functions
        """
        :param term:
        :param formula:
        :param prop:
        :param contributor:
        :param reference:
        :param min_measurement:
        :param max_measurement:3
        :param from_record: 
        :param per_page:
        :param data_set_id:
        :rtype: object
        """

        self.api_key = api_key
        self.json_data = []
        self.size = 1
        self.start = 0

        if self.api_key is None:
            client = CitrinationClient(os.environ['CITRINE_KEY'], 'http://citrination.com')
        else:
            client = CitrinationClient(self.api_key, 'http://citrination.com')

        while self.size > 0:
            self.data = client.search(term=term, formula=formula, property=property,
                                      contributor=contributor, reference=reference,
                                      min_measurement=min_measurement, max_measurement=max_measurement,
                                      from_record=self.start, per_page=100, data_set_id=data_set_id)
            self.size = len(self.data.json()['results'])
            self.start += self.size
            self.json_data.append(self.data.json()['results'])
            if self.size < 100:  # break out of last loop of results
                break
            time.sleep(3)
        self.hits = self.data.json()['hits']

    def to_pandas(self):

        non_meas_df = pd.DataFrame()  # df w/o measurement column
        meas_prop_df = pd.DataFrame()  # df w/only measurement.property columns
        meas_nonprop_df = pd.DataFrame()   # df w/o measurement.property columns
        meas_df = pd.DataFrame()   # df containing only measurement column
        units = {}  # dict for containing units
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_colwidth', -1)
        # pd.set_option('display.max_rows', 1000)

        counter = 0  # variable to keep count of sample hit and set indexes

        for page in tqdm(self.json_data):
            # df = pd.concat((json_normalize(hit) for hit in set))   # Useful tool for the future
            for hit in tqdm(page):
                counter += 1
                if 'sample' in hit.keys():
                    sample_value = hit['sample']
                    sample_normdf = json_normalize(sample_value)
                    # Make a DF of all non-'measurement' fields
                    non_meas_cols = [cols for cols in sample_normdf.columns if "measurement" not in cols]
                    non_meas_row = pd.DataFrame()
                    for col in non_meas_cols:
                        non_meas_row[col] = sample_normdf[col]
                    non_meas_row.index = [counter] * len(sample_normdf)
                    non_meas_df = non_meas_df.append(non_meas_row)
                    # Make a DF of the 'measurement' array
                    if 'measurement' in sample_value:
                        meas_normdf = json_normalize(sample_value['measurement'])
                        # Extract numbers of properties
                        for row, col in enumerate(meas_normdf['property.scalar']):
                            for item in col:
                                if 'value' in item:
                                    meas_normdf.xs(row)['property.scalar'] = item['value']
                                # TODO: ask Anubhav how to deal with these and rest of formats
                                elif 'minimum' in item and 'maximum' in item:
                                    meas_normdf.xs(row)['property.scalar'] = 'Minimum = ' + item[
                                        'minimum'] + ', ' + 'Maximum = ' + item['maximum']
                        # Make a DF of all non-'property' containing fields in 'measurement'
                        non_prop_cols = [meascols for meascols in meas_normdf.columns if "property" not in meascols]
                        non_prop_df = pd.DataFrame()
                        for col in non_prop_cols:
                            non_prop_df['measurement.' + col] = meas_normdf[col]
                        if len(non_prop_df) > 0:
                            non_prop_df.index = [counter] * len(meas_normdf)
                        meas_nonprop_df = meas_nonprop_df.append(non_prop_df)
                        # Pivot the DF to convert properties to columns
                        prop_df = meas_normdf.pivot(columns='property.name',
                                                    values='property.scalar')
                        prop_df.index = [counter] * len(meas_normdf)
                        meas_prop_df = meas_prop_df.append(prop_df)
                        m_df = pd.concat([non_prop_df,prop_df], axis=1)
                        meas_df = meas_df.append(m_df)
                        # Extracting units
                        # Check to avoid an error with databases that don't contain this field
                        if 'property.units' in meas_normdf.columns:
                            curr_units = dict(zip(meas_normdf['property.name'], meas_normdf['property.units']))
                        for prop in curr_units:
                            if prop not in units:
                                units[prop] = curr_units[prop]
        units_lst = [units]
        # df = pd.concat(
        #         [non_meas_df, meas_prop_df, meas_nonprop_df, pd.Series(units_lst, index=[1], name='property.units')],
        #         axis=1)
        df = pd.concat(
                [non_meas_df, meas_df, pd.Series(units_lst, index=[1], name='property.units')],
                axis=1)
        df.index.name = 'sample'
        return df

def get_mass(x):
        mass = 0
        el_amt = Composition(x).get_el_amt_dict()
        for el in el_amt:
            mass += Element(el).atomic_mass*el_amt[el]
        return mass

if __name__ == '__main__':
    # c = CitrineDataRetrieval(contributor='Carrico')
    # c = CitrineDataRetrieval(contributor='Citrine', term='Wikipedia', formula='PbTe')
    # c = CitrineDataRetrieval(contributor='aflow', formula='Si')
    # c = CitrineDataRetrieval(contributor='Lany', formula='Pb2Pd3Te2')
    # c = CitrineDataRetrieval(contributor='Citrine', term='NIST', formula='al2o3')
    # c = CitrineDataRetrieval(contributor='Gaultois', formula='pbte')
    # c = CitrineDataRetrieval(contributor='Harada', formula='li3v2p3o12')
    c = CitrineDataRetrieval(contributor='oqmd', formula='GaN')
    # c = CitrineDataRetrieval(formula='PbTe', contributor='TE design lab')
    # c = CitrineDataRetrieval(formula='PbTe', property='band gap')     # 'ValueError: shape indices do not match' error occurs with this query when 'concat' is used on two DFs with all rows having the same index but the DFs themselves have different number os rows, which happens with the PbTe sample from 'TE design lab' which has 17 properties but no non-'property' columns in its 'measurement', i.e. empty 'non_prop_df
    #  of empty DF of non-measurement properties (occurs with TE design lab sample)
    # print c.print_output()
    d = c.to_pandas()
    print d
    # c = Composition('Li3V2P3O12')
    e = Element('Pb')
    # print c.get_el_amt_dict()
    print e.atomic_mass
    print 'Elec = '+ str(e.X)
    print d['material.chemicalFormula'].apply(get_mass)
    print d.loc[d['material.chemicalFormula'] == 'GaN']
    # print d.query['material.chemicalFormula == GaN']
