import requests
from lxml import html
from pymatgen.io.cif import CifParser
from pymatgen.matproj.snl import StructureNL
from matminer.pyCookieCheat import chrome_cookies
from BeautifulSoup import BeautifulSoup

i = 0
j = 1402653

while i <= 0:
    try:
        sim_user_token = chrome_cookies('http://materials.springer.com')['sim-user-token']
        page = requests.get('http://materials.springer.com/isp/crystallographic/docs/sd_' + str(j),
                            cookies={'sim-user-token': sim_user_token})
        if page.raise_for_status() is None:  # Check if getting data from above was successful or now
            print 'Success at getting sd_' + str(j)
            i += 1
            parsed_body = html.fromstring(page.content)
            soup = BeautifulSoup(page.content)
            for a_link in parsed_body.xpath('//a/@href'):
                if '.cif' in a_link:
                    cif_link = a_link
                    res = requests.get('http://materials.springer.com' + cif_link,
                                       cookies={'sim-user-token': sim_user_token})
                    with open('ciffile.txt', 'w') as cif_file:
                        cif_file.write(res.content)
                    cif_struct = CifParser.from_string(res.content).get_structures()[0]
                    ref = soup.find('div', {'id': 'globalReference'}).find('div', 'accordion__bd')
                    data_dict = {'_globalReference': ''.join([(str(item)).strip() for item in ref.contents])}
                    print StructureNL(cif_struct, data=data_dict,
                                      authors=['Saurabh Bajaj <sbajaj@lbl.gov>', 'Anubhav Jain <ajain@lbl.gov>'])
    except Exception as e:
        print e
    j += 1
