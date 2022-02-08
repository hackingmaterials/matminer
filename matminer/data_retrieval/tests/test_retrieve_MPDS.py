import os

import httplib2

try:
    import ujson as json
except ImportError:
    import json

import unittest

from jsonschema import Draft4Validator, validate
from jsonschema.exceptions import ValidationError

from matminer.data_retrieval.retrieve_MPDS import MPDSDataRetrieval
from matminer.data_retrieval.tests.base import on_ci


class MPDSDataRetrievalTest(unittest.TestCase):
    def setUp(self):
        self.test_request = {
            "elements": "K-Ag",
            "classes": "iodide",
            "props": "heat capacity",
            "lattices": "cubic",
        }

        network = httplib2.Http()

        response, content = network.request("http://developer.mpds.io/mpds.schema.json")
        assert response.status == 200

        self.schema = json.loads(content)
        Draft4Validator.check_schema(self.schema)

    @unittest.skipIf(on_ci.upper() == "TRUE", "Bad Datasource-GHActions pipeline")
    @unittest.skipIf("MPDS_KEY" not in os.environ, "MPDS_KEY env var not set")
    def test_valid_answer(self):

        client = MPDSDataRetrieval()
        answer = client.get_data(self.test_request, fields={})

        try:
            validate(answer, self.schema)
        except ValidationError as e:
            self.fail(f"The item: \r\n\r\n {e.instance} \r\n\r\n has an issue: \r\n\r\n {e.context}")


if __name__ == "__main__":
    unittest.main()
