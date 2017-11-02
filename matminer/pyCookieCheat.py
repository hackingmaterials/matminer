#! /usr/bin/env python

"""
For most recent version: https://github.com/n8henrie/pycookiecheat

Use your browser's cookies to make retrieving data from login-protected sites easier.
Intended for use with Python Requests http://python-requests.org
Accepts a URL from which it tries to extract a domain. If you want to force the domain,
just send it the domain you'd like to use instead.
"""

from __future__ import division, unicode_literals, print_function

try:
    from pycookiecheat import chrome_cookies
except ImportError as ex:
    print(ex)

import requests

# sample usage given below
url = 'http://example.com/fake.html'
# for google chrome the cookie path has to be explicitly set
cookie_file = "/home/username/.config/google-chrome/Default/Cookies"
cookies = chrome_cookies(url, cookie_file=cookie_file)
r = requests.get(url, cookies=cookies)
