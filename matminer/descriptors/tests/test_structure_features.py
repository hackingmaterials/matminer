# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

from __future__ import unicode_literals

#import numpy as np
import unittest2 as unittest
#import os

#from pymatgen.analysis.structure_analyzer import VoronoiCoordFinder, \
#    solid_angle, contains_peroxide, RelaxationAnalyzer, VoronoiConnectivity, \
#    oxide_type, sulfide_type, OrderParameters, average_coordination_number, \
#    VoronoiAnalyzer
#from pymatgen.io.vasp.inputs import Poscar
#from pymatgen.io.vasp.outputs import Xdatcar
from pymatgen import Structure, Lattice
from pymatgen.util.testing import PymatgenTest

from matminer.descriptors.structure_features import get_order_parameters

class StructureFeaturesTest(PymatgenTest):
    def setUp(self):
        self.diamond = Structure(
                Lattice([[2.18898221, 0.0, 1.26380947],
                [0.72966074, 2.06379222, 1.26380947],
                [0.0, 0.0, 2.52761893]]), ["C", "C"],
                [[2.55381258125, 1.8058181925, 4.42333313625],
                [0.36483036875, 0.2579740275, 0.63190473375]],
                validate_proximity=False, to_unit_cell=False,
                coords_are_cartesian=True, site_properties=None)


