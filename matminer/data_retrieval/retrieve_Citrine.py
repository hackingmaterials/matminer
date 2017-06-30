from __future__ import absolute_import
from citrination_client import *
import os
import time
import pandas as pd
from tqdm import tqdm
from pandas.io.json import json_normalize


__author__ = 'Saurabh Bajaj <sbajaj@lbl.gov>'

"""
This package requires downloading an installing the citrination client:
https://github.com/CitrineInformatics/python-citrination-client

"""


class CitrineDataRetrieval:
    def __init__(self, api_key=None):
        """
        Args:
            api_key: (str) Your Citrine API key, or None if you've set the CITRINE_KEY environment variable

        Returns: None
        """
        api_key = api_key if api_key else os.environ['CITRINE_KEY']
        self.client = CitrinationClient(api_key, 'https://citrination.com')

    def get_value(self, dict_item):
        """
        Extract values from 'Property' objects

        Args:
            dict_item: 'Property' object

        Returns:
            - if 'value', returns string/float
            - if 'minimum' or 'maximum', returns string

        """
        # TODO: deal with rest of formats in a scalar object
        if 'value' in dict_item:
            return dict_item['value']
        elif 'minimum' in dict_item and 'maximum' in dict_item:
            return 'Minimum = {}, Maximum = {}'.format(dict_item['minimum'], dict_item['maximum'])

    def parse_scalars(self, scalar_column):
        """
        Parse scalar/single value items from a 'Property' column

        Args:
            scalar_column: 'Property' column with scalar objects

        Returns: column with extracted values

        """
        for row, col in enumerate(scalar_column):
            try:
                for i in col:
                    scalar_column.set_value(row, self.get_value(i))
            except TypeError:
                continue
        return scalar_column

    def parse_vectors(self, vector_column):
        """
        Parse vector/array value items from a 'Property' column

        Args:
            vector_column: 'Property' column with vector objects

        Returns: column with extracted arrays

        """
        vector_values = []
        for row, col in enumerate(vector_column):
            try:
                for i in col:
                    for j in i:
                        vector_values.append(self.get_value(j))
            except TypeError:
                continue
            vector_column.set_value(row, vector_values)
        return vector_column

    def parse_matrix(self, matrix_column):
        """
        Parse matrix/(array of array) value items from a 'Property' column

        Args:
            matrix_column: 'Property' column with array of array objects

        Returns: column with extracted array of arrays

        """
        matrix_values = []
        for row, col in enumerate(matrix_column):
            try:
                for i in col:
                    for j in i:
                        row_values = []
                        for k in j:
                            row_values.append(self.get_value(k))
                        matrix_values.append(row_values)
            except TypeError:
                continue
            matrix_column.set_value(row, matrix_values)
        return matrix_values

    def get_dataframe(self, formula=None, property=None, data_type=None, reference=None, min_measurement=None,
                      max_measurement=None, from_record=None, data_set_id=None, max_results=None, show_columns=None):
        """
        Gets data from Citrine in a dataframe format.
        See client docs at http://citrineinformatics.github.io/api-documentation/ for more details on these parameters.

        Args:
            formula: (str) filter for the chemical formula field; only those results that have chemical formulas that
                contain this string will be returned
            property: (str) name of the property to search for
            data_type: (str) 'EXPERIMENTAL'/'COMPUTATIONAL'/'MACHINE_LEARNING'; filter for properties obtained from
                experimental work, computational methods, or machine learning.
            reference: (str) filter for the reference field; only those results that have contributors that
                contain this string will be returned
            min_measurement: (str/num) minimum of the property value range
            max_measurement: (str/num) maximum of the property value range
            from_record: (int) index of the first record to return (indexed from 0)
            data_set_id: (int) id of the particular data set to search on
            max_results: (int) number of records to limit the results to
            show_columns: (list) list of columns to show from the resulting dataframe

        Returns: (object) Pandas dataframe object containing the results
        """

        json_data = []
        start = from_record if from_record else 0
        per_page = 100
        refresh_time = 3  # seconds to wait between search calls

        while True:
            if max_results and max_results < per_page:  # use per_page=max_results, eg: in case of max_results=68 < 100
                pif_query = PifQuery(system=SystemQuery(
                    chemical_formula=ChemicalFieldOperation(filter=ChemicalFilter(equal=formula)),
                    properties=PropertyQuery(name=FieldOperation(filter=Filter(equal=property)),
                                             value=FieldOperation(filter=Filter(min=min_measurement,
                                                                                max=max_measurement)),
                                             data_type=FieldOperation(filter=Filter(equal=data_type))
                                             ),
                    references=ReferenceQuery(doi=FieldOperation(filter=Filter(equal=reference)))),
                    include_datasets=[data_set_id], from_index=start, size=max_results)

            else:
                pif_query = PifQuery(system=SystemQuery(
                    chemical_formula=ChemicalFieldOperation(filter=ChemicalFilter(equal=formula)),
                    properties=PropertyQuery(name=FieldOperation(filter=Filter(equal=property)),
                                             value=FieldOperation(filter=Filter(min=min_measurement,
                                                                                max=max_measurement)),
                                             data_type=FieldOperation(filter=Filter(equal=data_type))
                                             ),
                    references=ReferenceQuery(doi=FieldOperation(filter=Filter(equal=reference)))),
                    include_datasets=[data_set_id], from_index=start, size=per_page)

            # Check if any results found
            if 'hits' not in self.client.search(pif_query).as_dictionary():
                raise KeyError('No results found!')

            data = self.client.search(pif_query).as_dictionary()['hits']
            size = len(data)
            start += size
            json_data.append(data)

            if max_results and len(json_data) * per_page > max_results:      # check if limit is reached
                json_data = json_data[:(max_results / per_page)]             # get first multiple of 100 records
                json_data.append(data[:max_results % per_page])              # get remaining records
                break
            if size < per_page:  # break out of last loop of results
                break

            time.sleep(refresh_time)

        non_prop_df = pd.DataFrame()  # df w/o measurement column
        prop_df = pd.DataFrame()  # df containing only measurement column

        counter = 0  # variable to keep count of sample hit and set indexes

        for page in json_data:
            # df = pd.concat((json_normalize(hit) for hit in set))   # Useful tool for the future

            for hit in tqdm(page):

                counter += 1          # Keep a count to appropriately index the rows

                if 'system' in hit.keys():       # Check if 'system' key exists, else skip
                    system_value = hit['system']
                    system_normdf = json_normalize(system_value)

                    # Make a DF of all non-'properties' fields
                    non_prop_cols = [cols for cols in system_normdf.columns if "properties" not in cols]
                    non_prop_row = pd.DataFrame()
                    for col in non_prop_cols:
                        non_prop_row[col] = system_normdf[col]
                    non_prop_row.index = [counter] * len(system_normdf)
                    non_prop_df = non_prop_df.append(non_prop_row)

                    # Make a DF of the 'properties' array
                    if 'properties' in system_value:
                        prop_normdf = json_normalize(system_value['properties'])

                        # Parse each type of property value
                        if 'scalars' in prop_normdf.columns:
                            self.parse_scalars(prop_normdf['scalars'])
                        if 'vectors' in prop_normdf.columns:
                            self.parse_vectors(prop_normdf['vectors'])
                        if 'matrices' in prop_normdf.columns:
                            self.parse_matrix(prop_normdf['matrices'])

                        # Get non-Null property values, and merge them into a new single column 'property_values'
                        value_cols = []
                        for col in prop_normdf.columns:
                            if col in ['scalars', 'vectors', 'matrices']:
                                value_cols.append(prop_normdf[col].dropna())
                        prop_normdf['property_values'] = pd.concat(value_cols)

                        # Pivot to make properties into columns
                        values_df = prop_normdf.pivot(columns='name', values='property_values')
                        values_df.index = [counter] * len(prop_normdf)
                        # Convert to float type whichever columns can be converted
                        values_df = values_df.apply(pd.to_numeric, errors='ignore')

                        # Making a single row DF of columns that do not contain property values
                        non_values_df = pd.DataFrame()
                        non_values_cols = []
                        for col in prop_normdf.columns:
                            if col not in ['name', 'scalars', 'vectors', 'matrices', 'property_values']:
                                non_values_cols.append(col)
                        for col in non_values_cols:
                            non_values_df[col] = prop_normdf[col]
                        if len(non_values_df) > 0:  # Do not index empty DF (non-value columns absent)
                            non_values_df.index = [counter] * len(prop_normdf)

                        # Concatenate values and non-values DF
                        prop_df = prop_df.append(pd.concat([values_df, non_values_df], axis=1))

        # Concatenate 'properties' and 'non-properties' dataframes
        df = pd.concat([non_prop_df, prop_df], axis=1)
        df.index.name = 'system'

        # Remove uninformative columns, such as 'category' and 'uid'
        df.drop(['category', 'uid'], axis=1, inplace=True)

        # Filter out columns not selected
        if show_columns:
            df = df[show_columns]

        return df
