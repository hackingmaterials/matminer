import unittest

from pymongo import MongoClient
from pymatgen.util.testing import PymatgenTest

from matminer.data_retrieval.retrieve_MongoDB import clean_projection, remove_ints, MongoDataRetrieval
from matminer.data_retrieval.tests.base import on_ci


class MongoDataRetrievalTest(PymatgenTest):
    def test_cleaned_projection(self):
        p = ["n.o.e", "n.o.e.l", "a.b", "a.b.c", "m", "m.b"]
        result = clean_projection(p)
        self.assertEqual(set(result), {"a.b", "m", "n.o.e"})

        p = ["d.x", "d.y", "d.z", "a.b.c", "a.b.d.e", "m.n.x", "m.l.x"]
        result = clean_projection(p)
        self.assertEqual(set(result), {"d", "a.b", "m"})

    def test_remove_ints(self):
        self.assertEqual(remove_ints("a.1"), "a")
        self.assertEqual(remove_ints("a.1.x"), "a.x")

    @unittest.skipIf(not on_ci, "MongoDataRetrievalTest configured only to run on CI by default")
    def test_get_dataframe(self):
        db = MongoClient("localhost", 27017, username="admin", password="password").test_db
        c = db.test_collection
        docs = [
            {
                "some": {"nested": {"result": 14.5}},
                "other": "notnestedresult",
                "final": 16.938475 + i,
                "array": [1.4, 5.6, 11.2, 1.1],
                "valid": True,
            }
            for i in range(5)
        ]

        docs[-1]["valid"] = False
        c.insert_many(docs)

        mdr = MongoDataRetrieval(c)

        df = mdr.get_dataframe(
            criteria={"valid": True}, properties=["some.nested.result", "other", "final", "array", "valid"]
        )

        self.assertTrue((df["some.nested.result"] == 14.5).all())
        self.assertTrue((df["other"] == "notnestedresult").all())

        floats = df["final"] != 16.938475
        self.assertTrue(floats.any() and not floats.all())

        self.assertArrayAlmostEqual(df["array"].iloc[0], [1.4, 5.6, 11.2, 1.1])
        self.assertTrue(df["valid"].all())

        c.drop()
