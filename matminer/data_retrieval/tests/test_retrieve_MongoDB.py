# coding: utf-8

from __future__ import division, unicode_literals, absolute_import

import unittest

from matminer.data_retrieval.retrieve_MongoDB import clean_projection


class MongoDataRetrievalTest(unittest.TestCase):
    def test_cleaned_projection(self):
        p = ['n.o.e', 'n.o.e.l', 'a.b', 'a.b.c', 'm', 'm.b']
        result = clean_projection(p)
        self.assertEqual(set(result), {'a.b', 'm', 'n.o.e'})

        p = ['d.x', 'd.y', 'd.z', 'a.b.c', 'a.b.d.e']
        result = clean_projection(p)
        self.assertEqual(set(result), {'d', 'a.b'})
