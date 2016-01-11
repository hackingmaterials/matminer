from citrination_client import CitrinationClient
import os
import time
import pandas as pd
from tqdm import tqdm
from pandas.io.json import json_normalize


class CitrineDataRetrieval:
    def __init__(self, api_key=None, term=None, formula=None, prop=None, contributor=None, reference=None,
                 min_measurement=None,
                 max_measurement=None, from_record=None, per_page=None, data_set_id=None):
        """
        :param term:
        :param formula:
        :param prop:
        :param contributor:
        :param reference:
        :param min_measurement:
        :param max_measurement:
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
            self.data = client.search(term=term, formula=formula, property=prop,
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

    def print_output(self):
        return self.json_data

    def to_pandas(self):

        dsi = pd.Series(name='data_set_id')
        material_df = pd.DataFrame()
        p_measurement_df = pd.DataFrame()
        np_measurement_df = pd.DataFrame()
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_colwidth', -1)
        pd.set_option('display.max_rows', 1000)

        counter = 0  # variable to keep count of sample hit and set indexes

        for set in tqdm(self.json_data):
            # df = pd.concat((json_normalize(hit) for hit in set))
            for hit in tqdm(set):
                counter += 1
                if 'sample' in hit.keys():
                    sample_value = hit['sample']
                    dsi.set_value(counter, sample_value['data_set_id'])
                    material_row = json_normalize(sample_value['material'])
                    material_row.index = [counter] * len(material_row)
                    material_df = material_df.append(material_row)
                    if 'measurement' in sample_value:
                        meas_normdf = json_normalize(sample_value['measurement'])
                        non_property_cols = [cols for cols in meas_normdf.columns if "property" not in cols]
                        # property_cols = [cols for cols in meas_normdf.columns if "property" in cols]
                        non_properties_df = pd.DataFrame()
                        for i in non_property_cols:
                            non_properties_df[i] = meas_normdf[i]
                        non_properties_df.index = [counter] * len(meas_normdf)
                        properties_df = meas_normdf.pivot(columns='property.name',
                                                          values='property.scalar')  # TODO: get property units
                        properties_df.index = [counter] * len(meas_normdf)
                        np_measurement_df = np_measurement_df.append(non_properties_df)
                        p_measurement_df = p_measurement_df.append(properties_df)
        df = pd.concat([material_df, np_measurement_df, p_measurement_df], axis=1)
        df.index.name = 'Sample'
        return df


# c = CitrineDataRetrieval(contributor='Carrico')
c = CitrineDataRetrieval(contributor='Lany', formula='PbTe')
# c = CitrineDataRetrieval(property='band gap', formula='PbTe')
# print c.print_output()
print c.to_pandas()
