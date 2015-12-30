from citrination_client import CitrinationClient
import pandas as pd


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
        self.data = client.search(term=self.term, formula=self.formula, property=self.property,
                                  contributor=self.contributor, reference=self.reference,
                                  min_measurement=self.min_measurement, max_measurement=self.max_measurement,
                                  from_record=self.from_record, per_page=self.per_page, data_set_id=self.data_set_id)
        self.json_data = self.data.json()

    def print_output(self):
        return self.data.json()

    def to_pandas(self):
        # return pd.DataFrame(self.data.json(), columns=self.data.json().keys())
        # self.data_json = json.dumps(self.json_data)
        return pd.io.json.json_normalize(self.json_data['results'])