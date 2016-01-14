"""
This package requires downloading an installing the citrination client:
https://github.com/CitrineInformatics/python-citrination-client

"""
from citrination_client import CitrinationClient
import os
import time
import pandas as pd
from tqdm import tqdm
from pandas.io.json import json_normalize


class CitrineDataRetrieval:
    def __init__(self, api_key=None):
        """
        :param api_key: (str) Your Citrine API key, or None if you've set the CITRINE_KEY environment variable
        """

        self.client = CitrinationClient(api_key, 'http://citrination.com') if api_key else CitrinationClient(os.environ['CITRINE_KEY'], 'http://citrination.com')

    def get_dataframe(self, term=None, formula=None, property=None, contributor=None, reference=None,
                 min_measurement=None, max_measurement=None, from_record=None, per_page=None, data_set_id=None):
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

        json_data = []
        start = 0
        per_page = 100

        while True:
            data = self.client.search(term=term, formula=formula, property=property,
                                      contributor=contributor, reference=reference,
                                      min_measurement=min_measurement, max_measurement=max_measurement,
                                      from_record=start, per_page=per_page, data_set_id=data_set_id)
            size = len(data.json()['results'])
            start += size
            json_data.append(data.json()['results'])
            if size < per_page:  # break out of last loop of results
                break
            time.sleep(3)

        hits = data.json()['hits']  # TODO: what is the point of this line of code?

        non_meas_df = pd.DataFrame()  # df w/o measurement column
        meas_prop_df = pd.DataFrame()  # df w/only measurement.property columns
        meas_nonprop_df = pd.DataFrame()   # df w/o measurement.property columns
        meas_df = pd.DataFrame()   # df containing only measurement column
        units = {}  # dict for containing units
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_colwidth', -1)
        # pd.set_option('display.max_rows', 1000)

        counter = 0  # variable to keep count of sample hit and set indexes

        # TODO: tqdm is probably not needed here unless you find that processing the data takes a long time. When I had mentioned to use tqdm, I thought it was possible to do so for the query itself
        for page in tqdm(json_data):
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


if __name__ == '__main__':
    # TODO: move these into an "Examples" file (either Python file or .ipynb). Don't leave the code here.
    CITRINE_KEY = None
    c = CitrineDataRetrieval(CITRINE_KEY)