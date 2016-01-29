import requests
from lxml import html

page = requests.get('http://materials.springer.com/isp/crystallographic/docs/sd_0456276')
print page.raise_for_status()      # Check if getting data from above was successful or now
print page.status_code == requests.codes.ok      # Check if getting data from above was successful or now

parsed_body = html.fromstring(page.content)
labels = parsed_body.xpath('//*[@id="general_information"]/div[1]/div/div/ul/li[1]/strong/text()')
print labels

for a_link in parsed_body.xpath('//a/@href'):
    if '.cif' in a_link:
        cif_link = a_link

res = requests.get('http://materials.springer.com' + cif_link)

with open('ciffile.txt', 'wb') as cif_file:
    cif_file.write(res.content)

cif_file.close()

# Grab links to all images
# images = parsed_body.xpath('//img/@src')
# if not images:
#     sys.exit("Found No Images")
# Convert any relative urls to absolute urls
# images = [urlparse.urljoin(page.url, url) for url in images]
# print 'Found %s images' % len(images)

