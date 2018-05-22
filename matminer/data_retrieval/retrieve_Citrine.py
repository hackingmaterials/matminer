from __future__ import absolute_import, division
from citrination_client import CitrinationClient, ChemicalFieldQuery, \
    ChemicalFilter, FieldQuery, PropertyQuery, Filter, ReferenceQuery, \
    PifSystemQuery, DatasetQuery, DataQuery, PifSystemReturningQuery
import os
import time
import pandas as pd
from matminer.data_retrieval.retrieve_base import BaseDataRetrieval
from tqdm import tqdm
from pandas.io.json import json_normalize
import numpy as np
from collections import Counter


__author__ = ['Saurabh Bajaj <sbajaj@lbl.gov>',
             'Alireza Faghaninia <alireza.faghaninia@gmail.com>']


def parse_scalars(scalars):
    return get_value(scalars[0])


def get_value(dict_item):
    # TODO: deal with rest of formats in a scalar object
    if "value" in dict_item:
        return dict_item["value"]
    elif "minimum" in dict_item and "maximum" in dict_item:
        return "Minimum = {}, Maximum = {}".format(dict_item["minimum"],
                                                   dict_item["maximum"])


class CitrineDataRetrieval(BaseDataRetrieval):
    """
    CitrineDataRetrieval is used to retrieve data from the Citrination database
    See API client docs at api_link below.
    """
    def __init__(self, api_key=None):
        """
        Args:
            api_key: (str) Your Citrine API key, or None if
                you've set the CITRINE_KEY environment variable
        """
        api_key = api_key if api_key else os.environ["CITRINE_KEY"]
        self.client = CitrinationClient(api_key, "https://citrination.com")

    def api_link(self):
        return "https://citrineinformatics.github.io/python-citrination-client/"

    def get_dataframe(self, criteria, properties=None, common_fields=None,
                      secondary_fields=False, print_properties_options=True):
        """
        Gets a Pandas dataframe object from data retrieved from
        the Citrine API.

        Args:
            criteria (dict): see get_data method for supported keys except
                prop; prop should be included in properties.
            properties ([str]): requested properties/fields/columns.
                For example, ["Seebeck coefficient", "Band gap"]. If unsure
                about the exact words, capitalization, etc try something like
                ["gap"] and "max_results": 3 and print_properties_options=True
                to see the exact options for this field
            common_fields ([str]): fields that are common to all the requested
                properties. Common example can be "chemicalFormula". Look for
                suggested common fields after a quick query for more info
            secondary_fields (bool): if True, fields not included in properties
                may be added to the output (e.g. references). Recommended only
                if len(properties)==1
            print_properties_options (bool): whether to print available options
                for "properties" and "common_fields" arguments.
        Returns: (object) Pandas dataframe object containing the results

        """
        common_fields = common_fields or []
        properties = properties or [None]
        if criteria.get("prop"):
            properties.append(criteria.pop("prop"))
            properties = list(set(properties))
        all_fields = []
        for prop_counter, requested_prop in enumerate(properties):
            jsons = self.get_data(**criteria, prop=requested_prop)
            non_prop_df = pd.DataFrame()  # df w/o measurement column
            prop_df = pd.DataFrame()  # df containing only measurement column
            counter = 0  # variable to keep count of sample hit and set indexes
            for hit in tqdm(jsons):
                counter += 1
                if "system" in hit.keys():  # Check if 'system' key exists, else skip
                    system_value = hit["system"]
                    system_normdf = json_normalize(system_value)
                    non_prop_cols = [cols for cols in system_normdf.columns
                                     if "properties" not in cols]
                    non_prop_row = pd.DataFrame()
                    for col in non_prop_cols:
                        non_prop_row[col] = system_normdf[col]
                    non_prop_row.index = [counter] * len(system_normdf)
                    non_prop_df = non_prop_df.append(non_prop_row)
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
                                    if isinstance(prop[prop_key], list) and len(prop[prop_key])>1:
                                        p_df[prop["name"] + "-" + prop_key] = np.nan
                                        p_df[prop["name"] + "-" + prop_key] = \
                                            p_df[prop["name"] + "-" + prop_key].astype(object)
                                    p_df.set_value(counter, prop["name"] + "-" + prop_key, prop[prop_key])
                        p_df.index = [counter]
                        prop_df = prop_df.append(p_df)
            df_prop = pd.concat([non_prop_df, prop_df], axis=1)
            if prop_counter==0:
                optcomcols = df_prop.columns.values
            else:
                optcomcols = list(set(optcomcols) & set(df_prop.columns))
            all_fields += list(df_prop.columns.values)
            if not secondary_fields:
                outcols = []
                for p in df_prop.columns.values:
                    if not requested_prop or requested_prop in p:
                        outcols.append(p)
                df_prop = df_prop[outcols + ["uid"] + common_fields]
            if prop_counter == 0:
                df = df_prop
            else:
                if "uid" in df and "uid" in df_prop:
                    uid = ["uid"]
                else:
                    uid = []
                try:
                    if len(uid+common_fields) > 0:
                        df = df.merge(df_prop, on=uid+common_fields, how="outer")
                    else:
                        df = df.join(df_prop, how="outer")
                except (TypeError, KeyError):
                    raise TypeError('Use scalar/string fields for common_fields'
                                    'common_fields among: {}'.format(optcomcols))
        uninformative_columns = ["category", "uid"]
        optcomcols = [c for c in optcomcols if c not in uninformative_columns]
        for col in uninformative_columns:
            if col in df:
                df = df.drop(col, axis=1)
        if print_properties_options:
            print("all available fields:\n{}".format(list(set(all_fields))))
            print("\nsuggested common fields:\n{}".format(optcomcols))
        return df

    def get_data(self, formula=None, prop=None, data_type=None,
                     reference=None, min_measurement=None, max_measurement=None,
                     from_record=None, data_set_id=None, max_results=None):
        """
        Gets raw api data from Citrine in json format. See api_link for more
        information on input parameters

        Args:
            formula: (str) filter for the chemical formula field; only those
                results that have chemical formulas that contain this string
                will be returned
            prop: (str) name of the property to search for
            data_type: (str) 'EXPERIMENTAL'/'COMPUTATIONAL'/'MACHINE_LEARNING';
                filter for properties obtained from experimental work,
                computational methods, or machine learning.
            reference: (str) filter for the reference field; only those
                results that have contributors that contain this string
                will be returned
            min_measurement: (str/num) minimum of the property value range
            max_measurement: (str/num) maximum of the property value range
            from_record: (int) index of first record to return (indexed from 0)
            data_set_id: (int) id of the particular data set to search on
            max_results: (int) number of records to limit the results to

        Returns: (list) of jsons/pifs returned by Citrine's API
        """

        json_data = []
        start = from_record if from_record else 0
        per_page = 100
        refresh_time = 3  # seconds to wait between search calls

        # Construct all of the relevant queries from input args
        formula_query = ChemicalFieldQuery(filter=ChemicalFilter(equal=formula))
        prop_query = PropertyQuery(name=FieldQuery(filter=Filter(equal=prop)),
                                   value=FieldQuery(filter=Filter(min=min_measurement,
                                                                  max=max_measurement)),
                                   data_type=FieldQuery(filter=Filter(equal=data_type)))
        ref_query = ReferenceQuery(doi=FieldQuery(filter=Filter(equal=reference)))

        system_query = PifSystemQuery(chemical_formula=formula_query,
                                      properties=prop_query,
                                      references=ref_query)
        dataset_query = DatasetQuery(id=Filter(equal=data_set_id))
        data_query = DataQuery(system=system_query, dataset=dataset_query)

        while True:
            # use per_page=max_results, eg: in case of max_results=68 < 100
            if max_results and max_results < per_page:
                pif_query = PifSystemReturningQuery(query=data_query,
                                                    from_index=start,
                                                    size=max_results)
            else:
                pif_query = PifSystemReturningQuery(query=data_query,
                                                    from_index=start,
                                                    size=per_page)

            # Check if any results found
            if "hits" not in self.client.search.pif_search(pif_query).as_dictionary():
                raise KeyError("No results found!")

            data = self.client.search.pif_search(pif_query).as_dictionary()["hits"]
            size = len(data)
            start += size
            json_data.extend(data)

            # check if limit is reached
            if max_results and len(json_data) > max_results:
                # get first multiple of 100 records
                json_data = json_data[:max_results]
                break
            if size < per_page:  # break out of last loop of results
                break
            time.sleep(refresh_time)
        return json_data
