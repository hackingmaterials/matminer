# coding: utf-8

from __future__ import division, unicode_literals, absolute_import

import unittest2 as unittest

from matminer.data_retrieval.retrieve_MongoDB import clean_projection

class MongoDataRetrievalTest(unittest.TestCase):
    def test_cleaned_projection(self):
        p1 = ['n.o.e', 'n.o.e.l', 'a.b', 'a.b.c', 'm', 'm.b']
        result = clean_projection(p1)
        self.assertEqual(set(result), set(['a.b', 'm', 'n.o.e']))
