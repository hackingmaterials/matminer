# coding: utf-8

from __future__ import division, unicode_literals, absolute_import
import unittest
from matminer.utils.flatten_dict import flatten_dict


class FlattenDictTest(unittest.TestCase):
    def test_flatten_nested_dict(self):
        # test basic functionality
        test1 = {"a": {"b": 1, "c": 2}}
        flattened = flatten_dict(test1)
        self.assertEqual(flattened["a.b"], 1)
        self.assertEqual(flattened["a.c"], 2)

        deep = {"a": {"b": {"c": {"d": 1}}}}
        deep_flat = flatten_dict(deep)
        self.assertEqual(deep_flat['a.b.c.d'], 1)

        # test array functionality
        test2 = {"a": {"b": (0, 1, 2), "c": 2}}
        flattened = flatten_dict(test2)
        self.assertEqual(flattened["a.b.0"], 0)
        self.assertEqual(flattened["a.b.2"], 2)

        flattened = flatten_dict(test2, unwind_arrays=False)
        self.assertEqual(flattened["a.b"], (0, 1, 2))


if __name__ == "__main__":
    unittest.main()
