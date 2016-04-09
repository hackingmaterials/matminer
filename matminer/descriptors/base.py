# coding: utf-8

from __future__ import division, print_function, unicode_literals, \
    absolute_import

"""
This module defines the base class
"""

import pandas as pd


__author__ = "Kiran Mathew"
__email__ = "kmathew@lbl.gov"


class BaseDescriptor(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def as_dict(self):
        return self.__dict__

    def as_frame(self):
        return pd.DataFrame(dict([(k,[v]) for k,v in self.as_dict().items() if k != "name"]),
                            index=[getattr(self, "name", "noname")])

    @staticmethod
    def from_dict(d):
        return BaseDescriptor(**d)
