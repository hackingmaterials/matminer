"""
The MIT License
Copyright (c) 2017, Evgeny Blokhin, Tilde Materials Informatics

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from __future__ import division
import os
import sys
import time
try: from urllib.parse import urlencode
except ImportError: from urllib import urlencode

import httplib2
import ujson as json

if not os.getenv('SKIP_MPDS_DEPS'):
    import jmespath
    import pandas as pd
    from ase import Atom
    from ase.spacegroup import crystal


class APIError(Exception):
    """Simple error handling"""
    def __init__(self, msg, code=0):
        self.msg = msg
        self.code = code
    def __str__(self):
        return repr(self.msg)

class MPDSDataRetrieval(object):
    """
    An example Python implementation
    of the API consumer for the MPDS platform,
    see http://developer.mpds.io

    Usage:
    $>export MPDS_KEY=...

    client = MPDSDataRetrieval()

    dataframe = client.get_dataframe({"formula":"SrTiO3", "props":"phonons"})

    *or*
    jsonobj = client.get_data(
        {"formula":"SrTiO3", "sgs": 99, "props":"atomic properties"},
        fields={
            'S':["entry", "cell_abc", "sg_n", "setting", "basis_noneq", "els_noneq"]
        }
    )

    *or*
    jsonobj = client.get_data({"formula":"SrTiO3"}, fields=[])
    """
    default_fields = {
        'S': [
            'phase_id',
            'chemical_formula',
            'sg_n',
            'entry',
            lambda: 'crystal structure',
            lambda: 'A'
        ],
        'P': [
            'sample.material.phase_id',
            'sample.material.chemical_formula',
            'sample.material.condition[0].scalar[0].value',
            'sample.material.entry',
            'sample.measurement[0].property.name',
            'sample.measurement[0].property.units',
            'sample.measurement[0].property.scalar'
        ],
        'C': [
            lambda: None,
            'title',
            lambda: None,
            'entry',
            lambda: 'phase diagram',
            'naxes',
            'arity'
        ]
    }
    default_titles = ['Phase', 'Formula', 'SG', 'Entry', 'Property', 'Units', 'Value']

    endpoint = "https://api.mpds.io/v0/download/facet"

    pagesize = 1000
    maxnpages = 100 # NB one hit may reach 50kB in RAM, consider pagesize*maxnpages*50kB free RAM
    chillouttime = 3 # NB please, do not use values < 3, because the server may burn out

    def __init__(self, api_key=None, endpoint=None):
        self.api_key = api_key if api_key else os.environ['MPDS_KEY']
        self.network = httplib2.Http()
        self.endpoint = endpoint or MPDSDataRetrieval.endpoint

    def _request(self, query, phases=[], page=0):
        phases = ','.join([str(int(x)) for x in phases]) if phases else ''

        response, content = self.network.request(
            uri=self.endpoint + '?' + urlencode({
                'q': json.dumps(query),
                'phases': phases,
                'page': page,
                'pagesize': MPDSDataRetrieval.pagesize
            }),
            method='GET',
            headers={'Key': self.api_key}
        )

        if response.status != 200:
            return {'error': 'HTTP error code %s' % response.status, 'code': response.status}
        try:
            content = json.loads(content)
        except:
            return {'error': 'Unreadable data obtained'}
        if content.get('error'):
            return {'error': content['error']}
        if not content['out']:
            return {'error': 'No hits', 'code': 1}

        return content

    def _massage(self, array, fields):
        if not fields:
            return array

        output = []

        for item in array:
            filtered = []
            for object_type in ['S', 'P', 'C']:
                if item['object_type'] == object_type:
                    for expr in fields.get(object_type, []):
                        if isinstance(expr, jmespath.parser.ParsedResult):
                            filtered.append(expr.search(item))
                        else:
                            filtered.append(expr)
                    break
            else:
                raise APIError("API error: unknown data type")

            output.append(filtered)

        return output

    def get_data(self, search, phases=[], fields=default_fields):
        """
        Retrieve data in JSON.
        JSON is expected to be valid against the schema
        http://developer.mpds.io/mpds.schema.json
        """
        output = []
        counter, hits_count = 0, 0
        fields = {
            key: [jmespath.compile(item) if isinstance(item, str) else item() for item in value]
            for key, value in fields.iteritems()
        } if fields else None

        while True:
            result = self._request(search, phases=phases, page=counter)
            if result['error']:
                raise APIError(result['error'], result.get('code', 0))

            if result['npages'] > MPDSDataRetrieval.maxnpages:
                raise APIError(
                    "Too much hits (%s > %s), please, be more specific" % \
                    (result['count'], MPDSDataRetrieval.maxnpages*MPDSDataRetrieval.pagesize),
                    1
                )
            assert result['npages'] > 0

            output.extend(self._massage(result['out'], fields))

            if hits_count and hits_count != result['count']:
                raise APIError("API error: hits count has been changed during the query")
            hits_count = result['count']

            if counter == result['npages'] - 1:
                break

            counter += 1
            time.sleep(MPDSDataRetrieval.chillouttime)

            sys.stdout.write("\r\t%d%%" % ((counter/result['npages']) * 100))
            sys.stdout.flush()

        if len(output) != hits_count:
            raise APIError("API error: collected and declared counts of hits differ")

        sys.stdout.write("\r\nGot %s hits\r\n" % hits_count)
        sys.stdout.flush()
        return output

    def get_dataframe(self, *args, **kwargs):
        """
        Retrieve data as a pandas dataframe.
        """
        columns = kwargs.get('columns')
        if columns:
            del kwargs['columns']
        else:
            columns = MPDSDataRetrieval.default_titles

        return pd.DataFrame(self.get_data(*args, **kwargs), columns=columns)

    @staticmethod
    def compile_crystal(datarow):
        """
        Helper method for processing
        the MPDS crystalline structures.
        NB crystalline structures are not retrieved by default,
        one needs to specify fields:
            cell_abc
            sg_n
            setting
            basis_noneq
            els_noneq
        """
        if not datarow or not datarow[-1]:
            return None

        cell_abc, sg_n, setting, basis_noneq, els_noneq = \
            datarow[-5], int(datarow[-4]), datarow[-3], datarow[-2], datarow[-1]

        atom_data = []
        setting = 2 if setting == '2' else 1

        for num, i in enumerate(basis_noneq):
            atom_data.append(Atom(els_noneq[num].encode('ascii'), tuple(i)))

        return crystal(
            atom_data,
            spacegroup=sg_n,
            cellpar=cell_abc,
            primitive_cell=True,
            setting=setting,
            onduplicates='replace' # NB here occupancies aren't currently considered
        )