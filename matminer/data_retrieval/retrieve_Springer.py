import requests
from lxml import html
from pymatgen.io.cif import CifParser
from pymatgen.matproj.snl import StructureNL
from matminer.pyCookieCheat import chrome_cookies

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
            for a_link in parsed_body.xpath('//a/@href'):
                if '.cif' in a_link:
                    cif_link = a_link
                    res = requests.get('http://materials.springer.com' + cif_link,
                                       cookies={'sim-user-token': sim_user_token})
                    with open('ciffile.txt', 'w') as cif_file:
                        cif_file.write(res.content)
                    a = CifParser.from_string(res.content).get_structures()[0]
                    print parsed_body.xpath('//*[@id="globalReference"]/div/div/text()')[0].strip()[:-1]
                    print StructureNL(a, authors=['Saurabh Bajaj <sbajaj@lbl.gov>', 'Anubhav Jain <ajain@lbl.gov>'])
    except Exception as e:
        print e
    j += 1
