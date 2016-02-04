import requests
from lxml import html
from pymatgen.io.cif import CifParser
from matminer.pyCookieCheat import chrome_cookies
import pymongo
import time

i = 0
j = 1402650

client = pymongo.MongoClient()
db = client['test-database']
collection = db['test-collection']
d = db['test-collection'].delete_many({})

cif_file = open('ciffile.txt', 'w')

while i < 20:
    try:
        sim_user_token = chrome_cookies('http://materials.springer.com')['sim-user-token']
        page = requests.get('http://materials.springer.com/isp/crystallographic/docs/sd_' + str(j),
                            cookies={'sim-user-token': sim_user_token})
        if page.raise_for_status() is None:  # Check if getting data from above was successful or now
            print 'Success at getting sd_' + str(j)
            i += 1
            parsed_body = html.fromstring(page.content)
            for a_link in parsed_body.xpath('//a/@href'):
                if '.cif' in a_link:
                    cif_link = a_link
                    res = requests.get('http://materials.springer.com' + cif_link,
                                       cookies={'sim-user-token': sim_user_token})
                    cif_file.write(res.content)
                    struct_dic = {'cif_string': res.content, 'webpage_str': page.content, "key": "sd_{}".format(j)}
                    try:
                        struct_dic['structure'] = CifParser.from_string(res.content).get_structures()[0].as_dict()
                    except:
                        struct_dic['structure'] = None
                        print("Could not parse structure for: sd_{}".format(j))
                    collection.insert(struct_dic)
    except Exception as e:
        print e
    j += 1
    time.sleep(0.5)

for record in collection.find():
    print record
