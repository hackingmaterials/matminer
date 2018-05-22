import httplib2
import ujson as json
import unittest
from jsonschema import validate, Draft4Validator
from jsonschema.exceptions import ValidationError
from matminer.data_retrieval.retrieve_MPDS import MPDSDataRetrieval


class MPDSDataRetrievalTest(unittest.TestCase):
    def setUp(self):
        self.test_request = {
            "elements": "K-Ag",
            "classes": "iodide",
            "props": "heat capacity",
            "lattices": "cubic"
        }

        network = httplib2.Http()

        response, content = network.request('http://developer.mpds.io/mpds.schema.json')
        assert response.status == 200

        self.schema = json.loads(content)
        Draft4Validator.check_schema(self.schema)

    def test_valid_answer(self):
        client = MPDSDataRetrieval()
        answer = client.get_data(self.test_request, fields={})

        try:
            validate(answer, self.schema)
        except ValidationError as e:
            self.fail(
                "The item: \r\n\r\n %s \r\n\r\n has an issue: \r\n\r\n %s" % (
                    e.instance, e.context
                )
            )


if __name__ == "__main__":
    unittest.main()
