import traceback
import requests
from lxml import html
from pymatgen.io.cif import CifParser
from matminer.pyCookieCheat import chrome_cookies
import pymongo
import time

if __name__ == '__main__':
    starting_page = 1
    total_pages = 100000000
    sleep_time = 0.5
    page_sleep_time = 1
    production_mode = True
    clear_production_db = False

    db_name = 'test_database' if not production_mode else 'springer'
    coll_name = 'test_collection' if not production_mode else 'pauling_file'

    print("Connecting to database...")
    client = pymongo.MongoClient()
    db = client[db_name]
    collection = db[coll_name]

    if not production_mode or clear_production_db:
        d = db[coll_name].delete_many({})

    print("Getting user token...")
    sim_user_token = chrome_cookies('http://materials.springer.com')['sim-user-token']

    print("Starting data collection")
    for page_no in range(starting_page, starting_page + total_pages):
        print("Starting page: {}".format(page_no))
        time.sleep(page_sleep_time)
        url = 'http://materials.springer.com/search?searchTerm=&pageNumber={}&propertyFacet=' \
              'crystal%20structure&datasourceFacet=sm_isp&substanceId='.format(page_no)
        result_page = requests.get(url)
        parsed_resbody = html.fromstring(result_page.content)
        for link in parsed_resbody.xpath('//a/@href'):
            if 'sd_' in link:
                sd_id = link[-10:]
                try:
                    link = 'http://materials.springer.com/' + str(link)
                    struct_page = requests.get(link, cookies={'sim-user-token': sim_user_token})
                    print 'Success at getting {}'.format(sd_id)
                    parsed_strucbody = html.fromstring(struct_page.content)
                    data_dict = {'webpage_str': struct_page.content, 'key': '{}'.format(sd_id), 'webpage_link': link}
                    for a_link in parsed_strucbody.xpath('//a/@href'):
                        if '.cif' in a_link:
                            cif_link = 'http://materials.springer.com' + a_link
                            res = requests.get(cif_link, cookies={'sim-user-token': sim_user_token})
                            data_dict['cif_string'] = res.content
                            data_dict['cif_link'] = cif_link
                            try:
                                data_dict['structure'] = CifParser.from_string(res.content).get_structures()[
                                    0].as_dict()
                            except:
                                print('! Could not parse structure for: {}'.format(sd_id))
                                print(traceback.format_exc())
                            break
                    if 'cif_string' not in data_dict:
                        print('!! Could not get CIF file for: {}'.format(sd_id))
                    collection.insert(data_dict)
                except:
                    print('! Could not fetch page for: http://materials.springer.com/{}'.format(link))
                    print(traceback.format_exc())
                time.sleep(sleep_time)

    # quick check
    print collection.find_one()
    print collection.find_one().keys()
    print collection.find_one()['key']
    print collection.count()
    print db['pauling_file'].find_one()
    print("FINISHED!")
