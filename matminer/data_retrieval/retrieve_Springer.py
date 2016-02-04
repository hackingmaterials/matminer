import traceback

import requests
from lxml import html
from pymatgen.io.cif import CifParser
from matminer.pyCookieCheat import chrome_cookies
import pymongo
import time


if __name__ == "__main__":
    i = 0
    sd_id = 1402650
    max_to_parse = 20
    sleep_time = 0.5
    clear_production_database = False
    testing_mode = True

    db_name = "test-database" if testing_mode == "testing" else "springer"
    coll_name = "test-collection" if testing_mode == "testing" else "pauling_file"

    client = pymongo.MongoClient()
    db = client[db_name]
    collection = db[coll_name]

    if testing_mode == "testing" or clear_production_database:
        d = db[coll_name].delete_many({})

    sim_user_token = chrome_cookies('http://materials.springer.com')['sim-user-token']

    while i < max_to_parse:
        try:
            page = requests.get('http://materials.springer.com/isp/crystallographic/docs/sd_' + str(sd_id),
                                cookies={'sim-user-token': sim_user_token})
            if page.raise_for_status() is None:  # Check if getting data from above was successful or now
                print 'Success at getting sd_{}'.format(sd_id)
                parsed_body = html.fromstring(page.content)
                data_dict = {"webpage_str": page.content, "key": "sd_{}".format(sd_id)}
                for a_link in parsed_body.xpath('//a/@href'):
                    if '.cif' in a_link:
                        cif_link = a_link
                        res = requests.get('http://materials.springer.com' + cif_link,
                                           cookies={'sim-user-token': sim_user_token})
                        data_dict = {'cif_string': res.content}
                        try:
                            data_dict['structure'] = CifParser.from_string(res.content).get_structures()[0].as_dict()
                        except:
                            data_dict['structure'] = None
                            print("! Could not parse structure for: sd_{}".format(sd_id))
                            print(traceback.format_exc())
                        break

                    else:
                        print("!! Could not get CIF file for: sd_{}".format(sd_id))
                collection.insert(data_dict)
                i += 1
        except:
            print(traceback.format_exc())

        sd_id += 1
        time.sleep(sleep_time)

    # quick check
    print collection.find_one()
    print("FINISHED!")