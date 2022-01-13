"""
Warning:
This retrieval class is to be deprecated in favor of the *mpds_client* library
`pip install mpds_client` (https://pypi.org/project/mpds-client),
which is fully compatible with matminer
"""

import os
import sys
import time
import warnings

import httplib2
from six.moves.urllib_parse import urlencode

from matminer.data_retrieval.retrieve_base import BaseDataRetrieval

try:
    import ujson as json
except ImportError:
    import json

import pandas as pd
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

try:
    import jmespath
except ImportError as ex:
    warnings.warn(str(ex))

use_ase = False
try:
    from ase import Atom
    from ase.spacegroup import crystal

    use_ase = True
except ImportError as ex:
    warnings.warn(str(ex))


__author__ = "Evgeny Blokhin <eb@tilde.pro>"
__copyright__ = "Copyright (c) 2017, Evgeny Blokhin, Tilde Materials Informatics"
__license__ = "MIT"


class APIError(Exception):
    """Simple error handling"""

    def __init__(self, msg, code=0):
        self.msg = msg
        self.code = code

    def __str__(self):
        return repr(self.msg)


class MPDSDataRetrieval(BaseDataRetrieval):
    """
    Retrieves data from Materials Platform for Data Science (MPDS).
    See api_link for more information.

    Usage:
    $>export MPDS_KEY=...

    client = MPDSDataRetrieval()

    dataframe = client.get_dataframe({"formula":"SrTiO3", "props":"phonons"})

    *or*
    jsonobj = client.get_data(
        {"formula":"SrTiO3", "sgs": 99, "props":"atomic properties"},
        fields={
            'S':["entry", "cell_abc", "sg_n", "basis_noneq", "els_noneq"]
        }
    )

    *or*
    jsonobj = client.get_data({"formula":"SrTiO3"}, fields={})

    If you use this data retrieval class, please additionally cite:
    Blokhin, E., Villars, P., 2018. The PAULING FILE Project and Materials
    Platform for Data Science: From Big Data Toward Materials Genome,
    in: Andreoni, W., Yip, S. (Eds.), Handbook of Materials Modeling:
    Methods: Theory and Modeling. Springer International Publishing, Cham,
    pp. 1-26. https://doi.org/10.1007/978-3-319-42913-7_62-2

    """

    default_properties = (
        "Phase",
        "Formula",
        "SG",
        "Entry",
        "Property",
        "Units",
        "Value",
    )

    endpoint = "https://api.mpds.io/v0/download/facet"

    pagesize = 1000
    maxnpages = 100  # NB one hit may reach 50kB in RAM, consider pagesize*maxnpages*50kB free RAM
    chillouttime = 2  # NB please, do not use values < 2

    def __init__(self, api_key=None, endpoint=None):
        """
        MPDS API consumer constructor

        Args:
            api_key: (str) The MPDS API key, or None if the MPDS_KEY envvar is set
            endpoint: (str) MPDS API gateway URL

        Returns: None
        """
        self.api_key = api_key if api_key else os.environ["MPDS_KEY"]
        self.network = httplib2.Http()
        self.endpoint = endpoint or MPDSDataRetrieval.endpoint

    def api_link(self):
        return "http://developer.mpds.io"

    def _request(self, query, phases=None, page=0):
        phases = ",".join([str(int(x)) for x in phases]) if phases else ""

        response, content = self.network.request(
            uri=self.endpoint
            + "?"
            + urlencode(
                {
                    "q": json.dumps(query),
                    "phases": phases,
                    "page": page,
                    "pagesize": MPDSDataRetrieval.pagesize,
                }
            ),
            method="GET",
            headers={"Key": self.api_key},
        )

        if response.status != 200:
            return {
                "error": "HTTP error code %s" % response.status,
                "code": response.status,
            }
        try:
            content = json.loads(content)
        except Exception:
            return {"error": "Unreadable data obtained"}
        if content.get("error"):
            return {"error": content["error"]}
        if not content["out"]:
            return {"error": "No hits", "code": 1}

        return content

    def _massage(self, array, fields):
        if not fields:
            return array

        output = []
        for item in array:
            filtered = []
            for object_type in ["S", "P", "C"]:
                if item["object_type"] == object_type:
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

    def get_data(self, criteria, phases=None, fields=None):
        """
        Retrieve data in JSON.
        JSON is expected to be valid against the schema
        at http://developer.mpds.io/mpds.schema.json

        Args:
            criteria (dict): Search query like {"categ_A": "val_A", "categ_B": "val_B"},
                documented at http://developer.mpds.io/#Categories
                example: criteria={"elements": "K-Ag", "classes": "iodide",
                                "props": "heat capacity", "lattices": "cubic"}
            phases (list): Phase IDs, according to the MPDS distinct phases concept
            fields (dict): Data of interest for C-, S-, and P-entries,
                e.g. for phase diagrams: {'C': ['naxes', 'arity', 'shapes']},
                documented at http://developer.mpds.io/#JSON-schemata

        Returns:
            List of dicts: C-, S-, and P-entries, the format is
            documented at http://developer.mpds.io/#JSON-schemata
        """

        default_fields = {
            "S": [
                "phase_id",
                "chemical_formula",
                "sg_n",
                "entry",
                lambda: "crystal structure",
                lambda: "A",
            ],
            "P": [
                "sample.material.phase_id",
                "sample.material.chemical_formula",
                "sample.material.condition[0].scalar[0].value",
                "sample.material.entry",
                "sample.measurement[0].property.name",
                "sample.measurement[0].property.units",
                "sample.measurement[0].property.scalar",
            ],
            "C": [
                lambda: None,
                "title",
                lambda: None,
                "entry",
                lambda: "phase diagram",
                "naxes",
                "arity",
            ],
        }

        fields = default_fields if fields is None else fields

        output = []
        phases = phases or []
        counter, hits_count = 0, 0
        fields = (
            {
                key: [jmespath.compile(item) if isinstance(item, str) else item() for item in value]
                for key, value in fields.items()
            }
            if fields
            else None
        )

        while True:
            result = self._request(criteria, phases=phases, page=counter)
            if result["error"]:
                raise APIError(result["error"], result.get("code", 0))

            if result["npages"] > MPDSDataRetrieval.maxnpages:
                raise APIError(
                    "Too much hits (%s > %s), please, be more specific"
                    % (
                        result["count"],
                        MPDSDataRetrieval.maxnpages * MPDSDataRetrieval.pagesize,
                    ),
                    1,
                )
            assert result["npages"] > 0

            output.extend(self._massage(result["out"], fields))

            if hits_count and hits_count != result["count"]:
                raise APIError("API error: hits count has been changed during the query")
            hits_count = result["count"]

            if counter == result["npages"] - 1:
                break

            counter += 1
            time.sleep(MPDSDataRetrieval.chillouttime)

            sys.stdout.write("\r\t%d%%" % ((counter / result["npages"]) * 100))
            sys.stdout.flush()

        if len(output) != hits_count:
            raise APIError("API error: collected and declared counts of hits differ")

        sys.stdout.write("\r\nGot %s hits\r\n" % hits_count)
        sys.stdout.flush()
        return output

    def get_dataframe(self, criteria, properties=default_properties, **kwargs):
        """
        Retrieve data as a Pandas dataframe.

        Args:
            criteria (dict): the same as criteria in get_data
            properties ([str]): list of properties/titles to be included
            **kwargs: other keyword arguments available in get_data

        Returns: (object) Pandas DataFrame object containing the results
        """
        return pd.DataFrame(self.get_data(criteria=criteria, **kwargs), columns=properties)

    @staticmethod
    def compile_crystal(datarow, flavor="pmg"):
        """
        Helper method for representing the MPDS crystal structures in two flavors:
        either as a Pymatgen Structure object, or as an ASE Atoms object.

        Attention! These two flavors are not compatible, e.g.
        primitive vs. crystallographic cell is defaulted,
        atoms wrapped or non-wrapped into the unit cell, etc.

        Note, that the crystal structures are not retrieved by default,
        so one needs to specify the fields while retrieval:
            - cell_abc
            - sg_n
            - basis_noneq
            - els_noneq
        e.g. like this: {'S':['cell_abc', 'sg_n', 'basis_noneq', 'els_noneq']}
        NB. occupancies are not considered.

        Args:
            datarow: (list) Required data to construct crystal structure:
                [cell_abc, sg_n, basis_noneq, els_noneq]
            flavor: (str) Either "pmg", or "ase"

        Returns:
            - if flavor is pmg, Pymatgen Structure object
            - if flavor is ase, ASE Atoms object
        """
        if not datarow or not datarow[-1]:
            return None
        cell_abc, sg_n, basis_noneq, els_noneq = datarow[-4], int(datarow[-3]), datarow[-2], datarow[-1]

        if flavor == "pmg":
            return Structure.from_spacegroup(sg_n, Lattice.from_parameters(*cell_abc), els_noneq, basis_noneq)

        elif flavor == "ase" and use_ase:
            atom_data = []
            for num, i in enumerate(basis_noneq):
                atom_data.append(Atom(els_noneq[num], tuple(i)))

            return crystal(
                atom_data,
                spacegroup=sg_n,
                cellpar=cell_abc,
                primitive_cell=True,
                onduplicates="replace",
            )

        else:
            raise APIError("Crystal structure treatment unavailable")

    def citations(self):
        return []
