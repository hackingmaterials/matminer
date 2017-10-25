from __future__ import absolute_import, division
from citrination_client import *
import os
import time
import pandas as pd
from tqdm import tqdm
from pandas.io.json import json_normalize
import numpy as np
from collections import Counter


__author__ = 'Saurabh Bajaj <sbajaj@lbl.gov>'


def parse_scalars(scalars):
    return get_value(scalars[0])


def get_value(dict_item):
    # TODO: deal with rest of formats in a scalar object
    if "value" in dict_item:
        return dict_item["value"]
    elif "minimum" in dict_item and "maximum" in dict_item:
        return "Minimum = {}, Maximum = {}".format(dict_item["minimum"], dict_item["maximum"])


class CitrineDataRetrieval:
    def __init__(self, api_key=None):
        """
        Args:
            api_key: (str) Your Citrine API key, or None if you've set the CITRINE_KEY environment variable

        Returns: None
        """
        api_key = api_key if api_key else os.environ["CITRINE_KEY"]
        self.client = CitrinationClient(api_key, "https://citrination.com")

    def get_api_data(self, formula=None, property=None, data_type=None, reference=None, min_measurement=None,
                      max_measurement=None, from_record=None, data_set_id=None, max_results=None):
        """
        Gets data from Citrine in a dataframe format.
        See client docs at http://citrineinformatics.github.io/api-documentation/ for more details on these parameters.

        Args:
            formula: (str) filter for the chemical formula field; only those results that have chemical formulas that
                contain this string will be returned
            property: (str) name of the property to search for
            data_type: (str) 'EXPERIMENTAL'/'COMPUTATIONAL'/'MACHINE_LEARNING';
                filter for properties obtained from experimental work, computational methods, or machine learning.
            reference: (str) filter for the reference field; only those results that have contributors that contain
                this string will be returned
            min_measurement: (str/num) minimum of the property value range
            max_measurement: (str/num) maximum of the property value range
            from_record: (int) index of the first record to return (indexed from 0)
            data_set_id: (int) id of the particular data set to search on
            max_results: (int) number of records to limit the results to

        Returns: (list) of jsons/pifs returned by Citrine's API
        """

        json_data = []
        start = from_record if from_record else 0
        per_page = 100
        refresh_time = 3  # seconds to wait between search calls

        while True:
            if max_results and max_results < per_page:  # use per_page=max_results, eg: in case of max_results=68 < 100
                pif_query = PifQuery(system=SystemQuery(
                    chemical_formula=ChemicalFieldQuery(filter=ChemicalFilter(equal=formula)),
                    properties=PropertyQuery(name=FieldQuery(filter=Filter(equal=property)),
                                             value=FieldQuery(filter=Filter(min=min_measurement,
                                                                            max=max_measurement)),
                                             data_type=FieldQuery(filter=Filter(equal=data_type))),
                    references=ReferenceQuery(doi=FieldQuery(filter=Filter(equal=reference)))),
                    include_datasets=[data_set_id], from_index=start, size=max_results)

            else:
                pif_query = PifQuery(system=SystemQuery(
                    chemical_formula=ChemicalFieldQuery(filter=ChemicalFilter(equal=formula)),
                    properties=PropertyQuery(name=FieldQuery(filter=Filter(equal=property)),
                                             value=FieldQuery(filter=Filter(min=min_measurement,
                                                                            max=max_measurement)),
                                             data_type=FieldQuery(filter=Filter(equal=data_type))),
                    references=ReferenceQuery(doi=FieldQuery(filter=Filter(equal=reference)))),
                    include_datasets=[data_set_id], from_index=start, size=per_page)

            # Check if any results found
            if "hits" not in self.client.search(pif_query).as_dictionary():
                raise KeyError("No results found!")

            data = self.client.search(pif_query).as_dictionary()["hits"]
            size = len(data)
            start += size
            json_data.extend(data)

            if max_results and len(json_data) > max_results:                 # check if limit is reached
                json_data = json_data[:max_results]             # get first multiple of 100 records
                json_data.extend(data[:max_results % per_page])              # get remaining records
                break
            if size < per_page:  # break out of last loop of results
                break

            time.sleep(refresh_time)

        return json_data

    def get_dataframe(self, json_lst, show_columns=None):
        """
        Converts list of json/pifs to a Pandas dataframe

        Args:
            json_lst: (list) of json/pifs
            show_columns: (list) list of columns to show from the resulting dataframe

        Returns: (object) Pandas dataframe object containing the results

        """
        non_prop_df = pd.DataFrame()  # df w/o measurement column
        prop_df = pd.DataFrame()  # df containing only measurement column

        counter = 0  # variable to keep count of sample hit and set indexes

        for hit in tqdm(json_lst):

            counter += 1  # Keep a count to appropriately index the rows

            if "system" in hit.keys():  # Check if 'system' key exists, else skip
                system_value = hit["system"]
                system_normdf = json_normalize(system_value)

                # Make a DF of all non-'properties' fields
                non_prop_cols = [cols for cols in system_normdf.columns if "properties" not in cols]
                non_prop_row = pd.DataFrame()
                for col in non_prop_cols:
                    non_prop_row[col] = system_normdf[col]
                non_prop_row.index = [counter] * len(system_normdf)
                non_prop_df = non_prop_df.append(non_prop_row)

                # Make a DF of the 'properties' array
                if "properties" in system_value:

                    p_df = pd.DataFrame()

                    # Rename duplicate property names in a record with progressive numbering
                    all_prop_names = [x["name"] for x in system_value["properties"]]

                    counts = {k: v for k, v in Counter(all_prop_names).items() if v > 1}

                    for i in reversed(range(len(all_prop_names))):
                        item = all_prop_names[i]
                        if item in counts and counts[item]:
                            all_prop_names[i] += "_" + str(counts[item])
                            counts[item] -= 1

                    # add each property, and its associated fields, as a new column
                    for p_idx, prop in enumerate(system_value["properties"]):

                        # Rename property name according to above duplicate numbering
                        prop["name"] = all_prop_names[p_idx]

                        if "scalars" in prop:
                            p_df.set_value(counter, prop["name"], parse_scalars(prop["scalars"]))
                        elif "vectors" in prop:
                            p_df[prop["name"]] = prop["vectors"]
                        elif "matrices" in prop:
                            p_df[prop["name"]] = prop["matrices"]

                        # parse all keys in the Property object except 'name', 'scalars', 'vectors', and 'matrices'
                        for prop_key in prop:

                            if prop_key not in ["name", "scalars", "vectors", "matrices"]:

                                # If value is a list of multiple items, set the cell to the entire list by first
                                # converting to object type, else results in a ValueError/IndexError
                                if type(prop[prop_key]) == list and len(prop[prop_key]) > 1:
                                    p_df[prop["name"] + "-" + prop_key] = np.nan
                                    p_df[prop["name"] + "-" + prop_key] = \
                                        p_df[prop["name"] + "-" + prop_key].astype(object)

                                p_df.set_value(counter, prop["name"] + "-" + prop_key, prop[prop_key])

                    p_df.index = [counter]
                    prop_df = prop_df.append(p_df)

        # Concatenate 'properties' and 'non-properties' dataframes
        df = pd.concat([non_prop_df, prop_df], axis=1)
        df.index.name = "system"

        # Remove uninformative columns, such as 'category' and 'uid'
        df.drop(["category", "uid"], axis=1, inplace=True)

        # Filter out columns not selected
        if show_columns:
            df = df[show_columns]

        return df
