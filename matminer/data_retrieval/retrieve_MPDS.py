
from __future__ import division
import os
import sys
import time
import warnings
try: from urllib.parse import urlencode
except ImportError: from urllib import urlencode

import httplib2
import ujson as json

try:
    import jmespath
    import pandas as pd
except ImportError:
    warnings.warn("Pandas dataframes and/or JSON querying unavailable")

use_pmg, use_ase = False, False
try:
    from pymatgen.core.structure import Structure
    from pymatgen.core.lattice import Lattice
    use_pmg = True
except ImportError: pass
try:
    from ase import Atom
    from ase.spacegroup import crystal
    use_ase = True
except ImportError: pass


if not use_pmg and not use_ase:
    warnings.warn("Crystal structure treatment unavailable")

__author__ = 'Evgeny Blokhin <eb@tilde.pro>'
__copyright__ = 'Copyright (c) 2017, Evgeny Blokhin, Tilde Materials Informatics'
__license__ = 'MIT'

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
        """
        MPDS API consumer constructor

        Args:
            api_key: (str) The MPDS API key, or None if the MPDS_KEY envvar is set
            endpoint: (str) MPDS API gateway URL

        Returns: None
        """
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
        at http://developer.mpds.io/mpds.schema.json

        Args:
            search: (dict) Search query like {"categ_A": "val_A", "categ_B": "val_B"},
                documented at http://developer.mpds.io/#Categories
            phases: (list) Phase IDs, according to the MPDS distinct phases concept
            fields: (dict) Data of interest for C-, S-, and P-entries,
                e.g. for phase diagrams: {'C': ['naxes', 'arity', 'shapes']},
                documented at http://developer.mpds.io/#JSON-schemata

        Returns:
            List of dicts: C-, S-, and P-entries, the format is
            documented at http://developer.mpds.io/#JSON-schemata
        """
        output = []
        counter, hits_count = 0, 0
        fields = {
            key: [jmespath.compile(item) if isinstance(item, str) else item() for item in value]
            for key, value in fields.items()
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
        Retrieve data as a Pandas dataframe.

        Args:
            search: (dict) Search query like {"categ_A": "val_A", "categ_B": "val_B"},
                documented at http://developer.mpds.io/#Categories
            phases: (list) Phase IDs, according to the MPDS distinct phases concept
            fields: (dict) Data of interest for C-, S-, and P-entries,
                e.g. for phase diagrams: {'C': ['naxes', 'arity', 'shapes']},
                documented at http://developer.mpds.io/#JSON-schemata
            columns: (list) Column names for Pandas dataframe

        Returns: (object) Pandas dataframe object containing the results
        """
        columns = kwargs.get('columns')
        if columns:
            del kwargs['columns']
        else:
            columns = MPDSDataRetrieval.default_titles

        return pd.DataFrame(self.get_data(*args, **kwargs), columns=columns)

    @staticmethod
    def compile_crystal(datarow, flavor='pmg'):
        """
        Helper method for representing the MPDS crystal structures in two flavors:
        either as a Pymatgen Structure object, or as an ASE Atoms object.

        Attention! These two flavors are not compatible, e.g.
        primitive vs. crystallographic cell is defaulted,
        atoms wrapped or non-wrapped into the unit cell etc.

        Note, that the crystal structures are not retrieved by default,
        so one needs to specify the fields while retrieval:
            - cell_abc
            - sg_n
            - setting
            - basis_noneq
            - els_noneq
        e.g. like this: {'S':['cell_abc', 'sg_n', 'setting', 'basis_noneq', 'els_noneq']}
        NB. occupancies are not considered.

        Args:
            datarow: (list) Required data to construct crystal structure:
                [cell_abc, sg_n, setting, basis_noneq, els_noneq]
            flavor: (str) Either "pmg", or "ase"

        Returns:
            - if flavor is pmg, returns Pymatgen Structure object
            - if flavor is ase, returns ASE Atoms object
        """
        if not datarow or not datarow[-1]:
            return None

        cell_abc, sg_n, setting, basis_noneq, els_noneq = \
            datarow[-5], int(datarow[-4]), datarow[-3], datarow[-2], datarow[-1]

        if flavor == 'pmg' and use_pmg:
            return Structure.from_spacegroup(
                sg_n,
                Lattice.from_parameters(*cell_abc),
                els_noneq,
                basis_noneq
            )

        elif flavor == 'ase' and use_ase:
            atom_data = []
            setting = 2 if setting == '2' else 1

            for num, i in enumerate(basis_noneq):
                atom_data.append(Atom(els_noneq[num], tuple(i)))

            return crystal(
                atom_data,
                spacegroup=sg_n,
                cellpar=cell_abc,
                primitive_cell=True,
                setting=setting,
                onduplicates='replace'
            )

        else: raise APIError("Crystal structure treatment unavailable")