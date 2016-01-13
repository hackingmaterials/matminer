import os
import time
import pandas as pd
from citrination_client import CitrinationClient
from pandas.io.json import json_normalize
from tqdm import tqdm


class CitrineDataRetrieval:
    def __init__(self, api_key=None, term=None, formula=None, property=None, contributor=None, reference=None,
                 min_measurement=None,
                 max_measurement=None, from_record=None, per_page=None, data_set_id=None):
        """
        :param term:
        :param formula:
        :param property:
        :param contributor:
        :param reference:
        :param min_measurement:
        :param max_measurement:
        :param from_record:
        :param per_page:
        :param data_set_id:
        :rtype: object
        """

        # TODO: It is unclear that these variables need to be stored inside the class. You can probably remove these
        # lines of code
        self.api_key = api_key
        self.term = term
        self.formula = formula
        self.property = property
        self.contributor = contributor
        self.reference = reference
        self.min_measurement = min_measurement
        self.max_measurement = max_measurement
        self.from_record = from_record
        self.per_page = per_page
        self.data_set_id = data_set_id

        # TODO: Need to describe setting CITRINE_KEY as an environment variable to the user
        if self.api_key is None:
            client = CitrinationClient(os.environ['CITRINE_KEY'], 'http://citrination.com')
        else:
            client = CitrinationClient(self.api_key, 'http://citrination.com')

        self.json_data = []
        self.size = 1
        self.start = 0

        while self.size > 0:
            self.data = client.search(term=self.term, formula=self.formula, property=self.property,
                                      contributor=self.contributor, reference=self.reference,
                                      min_measurement=self.min_measurement, max_measurement=self.max_measurement,
                                      from_record=self.start, per_page=100, data_set_id=self.data_set_id)
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

        # TODO: anytime you find yourself defining a dozen variables, you are likely doing something inefficiently.
        # In this case, just create a single dataframe object and append to it as needed.
        df = pd.DataFrame()
        dsi = pd.Series(name='data_set_id')
        chemForm = pd.Series(name='material.chemicalFormula')
        commonName = pd.Series(name='material.commonName')
        matCond = pd.Series(name='material.conditions')
        measpropname = pd.Series(name='measurement.property.name')
        measpropscalar = pd.Series(name='measurement.property.scalar')
        measpropunits = pd.Series(name='measurement.property.units')
        measCond = pd.Series(name='measurement.condition')
        measMeth = pd.Series(name='measurement.method')
        measdataType = pd.Series(name='measurement.dataType')
        measref = pd.Series(name='measurement.reference')
        measurement_df = pd.DataFrame()
        sampleRef = pd.Series(name='sample.reference')
        cont = pd.Series(name='contacts')
        lic = pd.Series(name='licenses')
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_colwidth', -1)
        counter = 0

        for set in tqdm(self.json_data):
            # df = pd.concat((json_normalize(hit) for hit in set))
            for hit in tqdm(set):
                counter += 1
                if 'sample' in hit.keys():
                    sample_value = hit['sample']
                    if 'data_set_id' in sample_value:
                        dsi.set_value(counter, sample_value['data_set_id'])
                    if 'material' in sample_value:
                        material_value = sample_value['material']
                        if 'chemicalFormula' in material_value:
                            chemForm.set_value(counter, material_value['chemicalFormula'])
                        if 'commonName' in material_value:
                            commonName.set_value(counter, material_value['commonName'])
                        if 'condition' in material_value:
                            matCond.set_value(counter, material_value['condition'])
                    if 'measurement' in sample_value:
                        measurement_normdf = json_normalize(sample_value['measurement'])
                        if 'property.name' in measurement_normdf.columns:
                            measpropname = measpropname.append(pd.Series(measurement_normdf['property.name'].tolist(),
                                                                         index=[counter] * len(measurement_normdf),
                                                                         name='measurement.property.name'))
                        #     # TODO: check why NOT having name here doesn't insert column names
                        if 'property.scalar' in measurement_normdf.columns:
                            measpropscalar = measpropscalar.append(
                                    pd.Series(measurement_normdf['property.scalar'].tolist(),
                                              index=[counter] * len(measurement_normdf),
                                              name='measurement.property.scalar'))
                        if 'property.units' in measurement_normdf.columns:
                            measpropunits = measpropunits.append(
                                    pd.Series(measurement_normdf['property.units'].tolist(),
                                              index=[counter] * len(measurement_normdf),
                                              name='measurement.property.units'))
                        if 'condition' in measurement_normdf.columns:
                            measCond = measCond.append(
                                    pd.Series(measurement_normdf['condition'].tolist(),
                                              index=[counter] * len(measurement_normdf),
                                              name='measurement.condition'))
                        if 'method' in measurement_normdf.columns:
                            measMeth = measMeth.append(
                                    pd.Series(measurement_normdf['method'].tolist(),
                                              index=[counter] * len(measurement_normdf),
                                              name='measurement.method'))
                        if 'dataType' in measurement_normdf.columns:
                            measdataType = measdataType.append(
                                    pd.Series(measurement_normdf['dataType'].tolist(),
                                              index=[counter] * len(measurement_normdf),
                                              name='measurement.dataType'))
                        if 'reference' in measurement_normdf.columns:
                            measref = measref.append(
                                    pd.Series(measurement_normdf['reference'].tolist(),
                                              index=[counter] * len(measurement_normdf),
                                              name='measurement.reference'))
                    if 'reference' in sample_value:
                        sampleRef.set_value(counter, sample_value['reference'])
                    if 'contact' in sample_value:
                        cont.set_value(counter, sample_value['contact'])
                    if 'license' in sample_value:
                        lic.set_value(counter, sample_value['license'])

        df = pd.concat(
                [dsi, chemForm, commonName, matCond, sampleRef, cont, lic, measpropname, measpropscalar, measpropunits,
                 measCond, measMeth, measdataType, measref], axis=1)
        df.index.name = 'Sample'
        print df[7:9]
        print df.loc[[7,9]]
        print df.iloc[7,5]
        return df

c = CitrineDataRetrieval(contributor='Carrico')
# c = CitrineDataRetrieval(contributor='Lany', formula='PbTe')
# # c = CitrineDataRetrieval(property='band gap', formula='PbTe')   # TODO: check why you receive an error (ValueError("cannot reindex from a duplicate axis") when running this query (problem with measCond and measref).
# # print c.print_output()
print c.to_pandas()
