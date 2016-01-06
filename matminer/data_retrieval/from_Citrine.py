from citrination_client import CitrinationClient
import os
import time
import pandas as pd
from tqdm import tqdm
from pandas.io.json import json_normalize


class CitrineDataRetrieval:
    def __init__(self, api_key=None, term=None, formula=None, property=None, contributor=None, reference=None, min_measurement=None,
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

        # TODO: It is unclear that these variables need to be stored inside the class. You can probably remove these lines of code
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

        # TODO: use tqdm to show progressbar
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
        c = self.json_data

        # TODO: why jenkins.json? Why dumping the results to a file when the user didn't request it?
        # with open('jenkins.json', 'w') as outfile:
        #    json.dump(c, outfile)

    def print_output(self):
        return self.json_data

    def to_pandas(self):

        # TODO: anytime you find yourself defining a dozen variables, you are likely doing something inefficiently. In this case, just create a single dataframe object and append to it as needed.
        data_set_id = []
        chemicalFormula = []
        commonName = []
        composition = []
        matdbid = []
        icsdid = []
        cif = []
        icsd_spacegroup = []
        final_spacegroup = []
        material_conditions = []
        measurement = []
        external_conditions = []
        method = []
        datatype = []
        contacts = []
        licenses = []
        reference = []
        df = pd.DataFrame()
        counter = 0
        dsi = pd.Series(name='data_set_id')
        cN = pd.Series(name='commonName')
        matCond = pd.DataFrame()
        measMent = pd.Series(name='measurement')
        sampleRef = pd.Series(name='reference')
        cont = pd.Series(name='contacts')
        lic = pd.Series(name='licenses')
        for set in tqdm(self.json_data):
            # df = pd.DataFrame.append(df, json_normalize(set))
            # df = pd.concat((json_normalize(hit) for hit in set))
            for hit in tqdm(set):
                counter += 1
                if 'sample' in hit.keys():
                    sample_value = hit['sample']
                    if 'data_set_id' in sample_value:
                        dsi.set_value(counter, sample_value['data_set_id'])
                        # data_set_id.append(sample_value['data_set_id'])
                    if 'material' in sample_value:
                        material_value = sample_value['material']
                        if 'chemicalFormula' in material_value:
                            chemicalFormula.append(material_value['chemicalFormula'])
                        if 'commonName' in material_value:
                            cN.set_value(counter, material_value['commonName'])
                            # commonName.append(material_value['commonName'])
                        if 'composition' in material_value:   # TODO: json_normalize the 'composition' object
                            composition.append(material_value['composition'])
                        if 'id' in material_value:
                            for id in material_value['id']:
                                if 'MatDB ID' in id.values():
                                    matdbid.append(id['value'])
                                    # continue               # TODO: check different databases (eg: Lany) to see if 'continue' statement is needed here and below
                                if 'ICSD ID' in id.values():
                                    icsdid.append(id['value'])
                                    # continue
                        if 'cif' in material_value:
                            cif.append(material_value['cif'])
                        if 'condition' in material_value:
                            condition_row = json_normalize(sample_value['material'], 'condition')
                            # condition_row = json_normalize(material_value['condition'], 'scalar', ['name'])        # TODO: Need to modify this for multiple conditions
                            matCond = matCond.append(condition_row)
                            # matCond.columns = ['material.condition.name', 'material.condition.scalar']
                            # matCond.set_index()
                            # matCond.set_value(counter, material_value['condition'])
                            # material_conditions.append(material_value['condition'])
        #                 #     stability_conditions = {}
        #                 #     for cond in material_value['condition']:
        #                 #         if 'Qhull stability' in cond.values():
        #                 #             stability_conditions['Qhull stability'] = cond['scalar'][0]['value']
        #                 #             continue
        #                 #         if 'ICSD space group' in cond.values():
        #                 #             icsd_spacegroup.append(cond['scalar'][0]['value'])
        #                 #             continue
        #                 #         if 'Final space group' in cond.values():
        #                 #             final_spacegroup.append(cond['scalar'][0]['value'])
        #                 #             continue
        #                 #     material_conditions.append(stability_conditions)

                        # if 'condition' in material_value:
                        #     df = json_normalize(hit['sample']['material'])
                            # df = pd.concat((json_normalize(cond) for cond in material_value['condition']))
                            # for cond in material_value['condition']:
            #                     # df_row = json_normalize(cond['scalar'])
            #                     # df = pd.DataFrame.append(df, df_row)

                    if 'measurement' in sample_value:
                        measMent.set_value(counter, sample_value['measurement'])
                        # measurement_values = sample_value['measurement']
            #             for measure in measurement_values:
            #                 df_row = json_normalize(measure['condition'])
            #                 df = pd.DataFrame.append(df, df_row)
                            #             if 'measurement' in sample_value:
                        # measurement_values = sample_value['measurement']
        #                 properties = {}
        #                 for measure in measurement_values:
        #                     if 'property' in measure:
        #                         properties[measure['property']['name']] = measure['property']['scalar'][0]['value'] + \
        #                                                                   measure['property']['units']
        #                     measurement.append(properties)
        #                     if 'condition' in measure:
        #                         external = {}
        #                         for ext_cond in measure['condition']:
        #                             external[ext_cond['name']] = ext_cond['scalar'][0]['value']
        #                         external_conditions.append(external)
        #                     if 'method' in measure:
        #                         method.append(measure['method'])
        #                     if 'dataType' in measure:
        #                         datatype.append(measure['dataType'])   # TODO: Need to verify this
        #                     if 'reference' in measure:
        #                         for item in measure['reference']:
        #                             reference.append(item)
        #                     if 'contact' in measure:
        #                         contacts.append(measure['contact'])
        #                     if 'license' in measure:
        #                         licenses.append(measure['license'])
                    if 'reference' in sample_value:
                        sampleRef.set_value(counter, sample_value['reference'])
        #                 reference_values = sample_value['reference']
        #                 for item in reference_values:
        #                     reference.append(item)
                    if 'contact' in sample_value:
                        cont.set_value(counter, sample_value['contact'])
                        # contacts.append(sample_value['contact'])
                    if 'license' in sample_value:
                        lic.set_value(counter, sample_value['license'])
                        # licenses.append(sample_value['license'])

        # dsi = pd.Series(data_set_id, name='data_set_id')
        # cF = pd.Series(chemicalFormula, name='chemicalFormula')
        # cN = pd.Series(commonName, name='commonName')
        # cmP = pd.Series(composition, name='composition')
        # matID = pd.Series(matdbid, name='MatDB ID')
        # icsdID = pd.Series(icsdid, name='ICSD ID')
        # CIF = pd.Series(cif, name='CIF')
        # matCond = pd.Series(material_conditions, name='material.conditions')
        # print material_conditions
        # matCond.columns = ['material.condition.name', 'material.condition.scalar']

        df1 = pd.concat([dsi, cN, measMent, sampleRef, cont, lic], axis=1)
        df = pd.concat([df1, matCond])
        return df

        #
        #
        #             if 'material' in sample_value:
        #                 material_value = sample_value['material']
        #                 # return json_normalize(material_value)
        #
        #
        #
        #

c = CitrineDataRetrieval(contributor='Carrico')
print c.print_output()
print c.to_pandas()

