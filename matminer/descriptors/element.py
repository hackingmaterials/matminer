# coding: utf-8

from __future__ import division, print_function, unicode_literals, \
    absolute_import

"""
This module defines element descriptor
"""

import pandas as pd

from matminer.descriptors.base import BaseDescriptor


__author__ = "Kiran Mathew"
__email__ = "kmathew@lbl.gov"


class ElementDescriptor(BaseDescriptor):
    """
    Descriptor for atomic elements
    """

    @staticmethod
    def from_element(element, feature_list):
        """
        return descriptor from element

        Args:
            element (Element): pymatgen element
            feature_list (list): list of property names

        Returns:
            BaseDescriptor
        """
        d = dict([(feature, getattr(element, feature, None))
                                for feature in feature_list])
        d.update({"name":element.name})
        return BaseDescriptor.from_dict(d)

    @staticmethod
    def frame_from_composition(comp, feature_list):
        """
        return pandas dataframe for each element in the composition

        Args:
            comp (Composition): pymatgen composition
            feature_list (list): list of property names

        Returns:
            DataFrame
        """
        frames = []
        for el in comp.elements:
            eld = ElementDescriptor.from_element(el, feature_list)
            frames.append(eld.as_frame())
        return pd.concat(frames)
