import requests
from lxml import html

i = 0
j = 1402650
while i <= 25000:
    try:
        page = requests.get('http://materials.springer.com/isp/crystallographic/docs/sd_' + str(j))
        if page.raise_for_status() is None:      # Check if getting data from above was successful or now
            print 'Success at getting sd_' + str(j)
            i += 1
            parsed_body = html.fromstring(page.content)
            for a_link in parsed_body.xpath('//a/@href'):
                if '.cif' in a_link:
                    cif_link = a_link
                    res = requests.get('http://materials.springer.com' + cif_link)
                    with open('ciffile.txt', 'a') as cif_file:
                        cif_file.write(res.content)
    except Exception as e:
        print e
        print 'Error in getting sd_' + str(j)
    j += 1


cif_file.close()

# print page.status_code == requests.codes.ok      # Check if getting data from above was successful or now

# labels = parsed_body.xpath('//*[@id="general_information"]/div[1]/div/div/ul/li[1]/strong/text()')
# print labels

# Grab links to all images
# images = parsed_body.xpath('//img/@src')
# if not images:
#     sys.exit("Found No Images")
# Convert any relative urls to absolute urls
# images = [urlparse.urljoin(page.url, url) for url in images]
# print 'Found %s images' % len(images)
