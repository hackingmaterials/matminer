from citrination_client import CitrinationClient
import pandas as pd
import time
import json


class CitrineDataRetrieval:
    def __init__(self, term=None, formula=None, property=None, contributor=None, reference=None, min_measurement=None,
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

        client = CitrinationClient('hpqULmumJMAsqvk8VtifQgtt', 'http://citrination.com')
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
            if self.size < 100:           # break out of last loop of results
                break
            time.sleep(3)
        self.hits = self.data.json()['hits']
        c = self.json_data
        with open('jenkins.json', 'w') as outfile:
            json.dump(c, outfile)

    def print_output(self):
        return self.json_data

    def to_pandas(self):
        # for set in self.json_data:
        #    for hit in set:
        # return pd.read_json('jenkins.json')
        # return pd.DataFrame(self.data.json(), columns=self.data.json().keys())
        # self.data_json = json.dumps(self.json_data)
        return pd.io.json.json_normalize(self.json_data)

