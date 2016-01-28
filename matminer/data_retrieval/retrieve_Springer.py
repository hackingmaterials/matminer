from lxml import html
import requests

page = requests.get('http://materials.springer.com/isp/crystallographic/docs/sd_0456276')
tree = html.fromstring(page.content)

print type(page)
print page.status_code == requests.codes.ok
print len(page.text)
print page.text[:500]

print tree

cif = tree.xpath('//*[@id="action-download-cif-link"]')
print cif

labels = tree.xpath('//*[@id="general_information"]/div[1]/div/div/ul/li[1]/strong/text()')
print labels