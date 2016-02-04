import requests
from bs4 import BeautifulSoup
from lxml import html
from pymatgen.io.cif import CifParser
from pymatgen.matproj.snl import StructureNL
from matminer.pyCookieCheat import chrome_cookies
import pymatgen
import pymongo

i = 0
j = 1402653

client = pymongo.MongoClient()
db = client['test-database']
collection = db['test-collection']
print db

while i <= 4:
    try:
        sim_user_token = chrome_cookies('http://materials.springer.com')['sim-user-token']
        page = requests.get('http://materials.springer.com/isp/crystallographic/docs/sd_' + str(j),
                            cookies={'sim-user-token': sim_user_token})
        if page.raise_for_status() is None:  # Check if getting data from above was successful or now
            print 'Success at getting sd_' + str(j)
            i += 1
            parsed_body = html.fromstring(page.content)
            # soup = BeautifulSoup(page.content, 'lxml')
            for a_link in parsed_body.xpath('//a/@href'):
                if '.cif' in a_link:
                    cif_link = a_link
                    res = requests.get('http://materials.springer.com' + cif_link,
                                       cookies={'sim-user-token': sim_user_token})
                    # with open('ciffile.txt', 'w') as cif_file:
                    #     cif_file.write(res.content)
                    # cif_struct = CifParser.from_string(res.content).get_structures()[0]

                    # geninfo = soup.find('div', {'id': 'general_information'})
                    # print geninfo.get_text()
                    # for i in soup.findAll('li', 'data-list__item'):
                    #     print i.contents[0].strip()
                    # print ''.join([(str(item)).strip() for item in geninfo.contents])
                    # buyers = parsed_body.xpath('//li[@class="data-list__item"]/text()')
                    # sellers = parsed_body.xpath('//li[@class="data-list__item"]/span/text()')
                    # print buyers
                    # print sellers
                    # geninfo = soup.findAll('li', 'data-list__item')
                    # print geninfo.contents
                    # ref = soup.find('div', {'id': 'globalReference'}).find('div', 'accordion__bd')
                    # data_dict = {'_globalReference': ''.join([(str(item)).strip() for item in ref.contents]),
                    #              '_entireWebpage': soup.get_text(), '_cif': res.content}
                    # print StructureNL(cif_struct, data=data_dict,
                    #                   authors=['Saurabh Bajaj <sbajaj@lbl.gov>', 'Anubhav Jain <ajain@lbl.gov>'])
                    struct_dic = {'cif_string': res.content, 'webpage_str': page.content}
                    try:
                        struct_dic['structure'] = CifParser.from_string(res.content).get_structures()[0].as_dict()
                    except:
                        print("Could not parse structure for: sd_{}".format(j))
                    # print struct_dic
                    # print struct_dic.keys()
                    collection.insert(struct_dic)
                    # soup = BeautifulSoup(struct_dic['webpage_str'], 'lxml')
                    # print soup.find('div', {'id': 'globalReference'}).find('div', 'accordion__bd')
                    # print CifParser.from_string(struct_dic['cif_string']).get_structures()[0]
                    # print pymatgen.Structure.from_dict(struct_dic['structure'])
    except Exception as e:
        print e
    j += 1

for record in collection.find():
    print record
d = db['test-collection'].delete_many({})
print d.deleted_count
