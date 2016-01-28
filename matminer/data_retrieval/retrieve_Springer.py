import requests
from lxml import html
import sys
import urlparse

page = requests.get('http://materials.springer.com/isp/crystallographic/docs/sd_0456276')
print page.raise_for_status()
parsed_body = html.fromstring(page.content)

print parsed_body

print type(page)
print page.status_code == requests.codes.ok
print len(page.text)
print page.text[:500]

cif = parsed_body.xpath('//*[@id="action-download-cif-link"]')
print cif

labels = parsed_body.xpath('//*[@id="general_information"]/div[1]/div/div/ul/li[1]/strong/text()')
print labels

# Grab links to all images
images = parsed_body.xpath('//img/@src')
if not images:
    sys.exit("Found No Images")

# Convert any relative urls to absolute urls
images = [urlparse.urljoin(page.url, url) for url in images]
print 'Found %s images' % len(images)

# Only download first 10
for url in images[0:10]:
    r = requests.get(url)
    f = open('%s' % url.split('/')[-1], 'w')
    f.write(r.content)
    f.close()
