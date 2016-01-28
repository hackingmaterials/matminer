from lxml import html
import requests

page = requests.get('http://materials.springer.com/isp/crystallographic/docs/sd_0456276')
tree = html.fromstring(page.content)

print tree

cif = tree.xpath('//*[@id="action-download-cif-link"]')
print cif