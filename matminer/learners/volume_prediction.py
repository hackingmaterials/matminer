import os
import re
from pymatgen import MPRester, Element, Composition
from collections import defaultdict, namedtuple
from pymatgen.analysis.structure_analyzer import VoronoiCoordFinder
from sklearn.metrics import mean_squared_error
import pickle
import numpy as np
import pandas as pd

mpr = MPRester()

module_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(module_dir, 'data_bondlengths')


class VolumePredictor(object):
    """
    Class to predict the volume of a given structure, based on averages of bond lengths of given input structures.
    """

    def __init__(self):
        self.bond_lengths = defaultdict(list)
        self.avg_bondlengths = {}

    def get_data(self, nelements, e_above_hull):
        """
        Get data from MP that is to be included in the database

        :param nelements: (int) maximum number of elements a material can have that is to be included in the dataset
        :param e_above_hull: (float) energy above hull as defined by MP (in eV/atom)
        :return: (namedtuple) of lists of task_ids, structures, and volumes
        """
        data = namedtuple('data', 'task_ids structures volumes')
        criteria = {'nelements': {'$lte': nelements}}
        mp_results = mpr.query(criteria=criteria, properties=['task_id', 'e_above_hull', 'structure'])
        mp_ids = []
        mp_structs = []
        mp_vols = []
        for i in mp_results:
            if i['e_above_hull'] < e_above_hull:
                mp_ids.append(i['task_id'])
                mp_structs.append(i['structure'])
                mp_vols.append(i['structure'].volume)
        return data(task_ids=mp_ids, structures=mp_structs, volumes=mp_vols)

    def get_bondlengths(self, structure):
        """
        Get all bond lengths in a structure by bond type.

        :param structure: pymatgen structure object
        :return: (defaultdict(list)) with bond types in the format 'A-B' as keys, and bond lengths as values
        """
        bondlengths = defaultdict(list)
        bondlengths_detailed = defaultdict(list)
        sites_dist = namedtuple('sites_dist', 'sites dist')
        min_bonds = {}
        min_sites_dist = namedtuple('min_sites_dist', 'min_sites min_dist')
        for site_idx, site in enumerate(structure.sites):
            try:
                voronoi_sites = VoronoiCoordFinder(structure).get_coordinated_sites(site_idx, tol=0.5)
            except RuntimeError as r:
                print 'Error for site {} in {}: {}'.format(site, structure.composition, r)
                continue
            except ValueError as v:
                print 'Error for site {} in {}: {}'.format(site, structure.composition, v)
                continue
            site_element_str = ''.join(re.findall(r'[A-Za-z]', site.species_string))
            for vsite in voronoi_sites:
                s_dist = np.linalg.norm(vsite.coords - site.coords)
                # TODO: avoid "continue" statements if possible-it is dead simply to write this code without "continue"
                if s_dist < 0.1:
                    continue
                vsite_element_str = ''.join(re.findall(r'[A-Za-z]', vsite.species_string))
                bond = '-'.join(sorted([site_element_str, vsite_element_str]))
                bondlengths[bond].append(s_dist)
                bondlengths_detailed[bond].append(sites_dist(sites=[site, vsite], dist=s_dist))
        for bl in bondlengths:
            print bl, min(bondlengths[bl])
        for bondtype in bondlengths_detailed:
            min_dist = bondlengths_detailed[bondtype][0].dist
            min_sites = bondlengths_detailed[bondtype][0].sites
            for site_pair in bondlengths_detailed[bondtype]:
                if site_pair.dist < min_dist:
                    min_dist = site_pair.dist
                    min_sites = site_pair.sites
            min_bonds[bondtype] = min_sites_dist(min_sites=min_sites, min_dist=min_dist)
        print min_bonds
        return bondlengths

    def fit(self, structures, volumes, ids=None):
        """
        Given a set of input structures, it stores bond lengths in a defaultdict(list) and average bond lengths in a
        dictionary by bond type (keys).

        :param structures: (list) list of pymatgen structure objects
        :param volumes: (list) corresponding list of volumes of structures
        :param ids: (list) corresponding list of MP task_ids. Default=None
        """
        for idx, struct in enumerate(structures):
            struct_bls = self.get_bondlengths(struct)
            for bond in struct_bls:
                if ids is not None:      # TODO: add else statement
                    # self.bond_lengths[bond].extend(struct_bls[bond])         # Use all bls
                    self.bond_lengths[bond].append(struct_data(bond_length=min(struct_bls[bond]), task_id=ids[idx]))  # Only use minimum bl for each bond type
        for bond in self.bond_lengths:
            total = 0
            for item in self.bond_lengths[bond]:
                total += item.bond_length
            self.avg_bondlengths[bond] = total / len(self.bond_lengths[bond])

    def get_rmse(self, structure_bls):
        """
        Calculates root mean square error between bond lengths in a structure and the average expected bond lengths.

        :param structure_bls: defaultdict(list) of bond lengths as output by the function "get_bondlengths()"
        :return: (float) root mean square error
        """
        y_actual = []
        y_avg = []
        for bond in structure_bls:
            if bond in self.avg_bondlengths:
                y_avg.extend([self.avg_bondlengths[bond]]*len(structure_bls[bond]))
            else:
                print 'Warning missing bond length for {}'.format(bond)
                el1, el2 = bond.split("-")
                r1 = float(Element(el1).atomic_radius)
                r2 = float(Element(el2).atomic_radius)
                y_avg.extend([(r1+r2)]*len(structure_bls[bond]))
            y_actual.extend([min(structure_bls[bond])]*len(structure_bls[bond]))
        return mean_squared_error(y_actual, y_avg) ** 0.5

    def predict(self, structure):
        """
        Predict volume of a given structure based on rmse against average bond lengths from a set of input structures.
        Note: run this function after initializing the "self.avg_bondlengths" variable, through either of the functions
        fit() or get_avg_bondlengths().

        :param structure: (structure) pymatgen structure object to predict the volume of
        :return: (tuple) predited volume (float) and its rmse (float)
        """
        prediction_results = namedtuple('prediction_results', 'volume rmse')
        starting_volume = structure.volume
        predicted_volume = starting_volume
        min_rmse = self.get_rmse(self.get_bondlengths(structure))
        min_percentage = 100
        for i in range(80, 121):
            test_volume = (i * 0.01) * starting_volume
            structure.scale_lattice(test_volume)
            test_rmse = self.get_rmse(self.get_bondlengths(structure))
            if test_rmse < min_rmse:
                min_rmse = test_rmse
                predicted_volume = test_volume
                min_percentage = i
        if min_percentage < 85 or min_percentage > 115:
            return self.predict(structure)
        else:
            return prediction_results(volume=predicted_volume, rmse=min_rmse)

    def save_avg_bondlengths(self, filename):
        """
        Save the class variable "self.avg_bondlengths" calculated by fit().

        :param filename: name of file to store to
        """
        with open(os.path.join(data_dir, filename), 'w') as f:
            pickle.dump(self.avg_bondlengths, f, pickle.HIGHEST_PROTOCOL)

    def save_bondlengths(self, filename):
        """
        Save the class variable "self.bond_lengths" calculated by fit().

        :param filename: name of file to store to
        """
        with open(os.path.join(data_dir, filename), 'w') as f:
            pickle.dump(self.bond_lengths, f, pickle.HIGHEST_PROTOCOL)

    def get_avg_bondlengths(self, filename):
        """
        Extract the saved average bond lengths and save them in the class variable "self.avg_bondlengths".

        :param filename: name of file to extract average bond lengths from
        :return:
        """
        with open(os.path.join(data_dir, filename), 'r') as f:
            self.avg_bondlengths = pickle.load(f)

    def save_predictions(self, structure_data):
        """
        Save results of the predict() function in a dataframe.

        :param structure_data: (namedtuple) of lists of task_ids, structures, and volumes, as output by get_data()
        :return:
        """
        df = pd.DataFrame()
        for idx, structure in enumerate(structure_data.structures):
            try:
                b = pv.predict(structure)
            except RuntimeError as r:
                print r
                continue
            except ValueError as v:
                print v
                continue
            df = df.append({'task_id': structure_data.task_ids[idx],
                            'reduced_cell_formula': Composition(structure.composition).reduced_formula,
                            'actual_volume': structure.volume, 'predicted_volume': b.volume}, ignore_index=True)
        df.to_pickle(os.path.join(data_dir, 'test.pkl'))
        # print pd.read_pickle('test.pkl')
        # df.plot(x='actual_volume', y='predicted_volume', kind='scatter')
        # plt.show()


if __name__ == '__main__':
    struct_data = namedtuple('struct_data', 'bond_length task_id')
    # mpids = ['mp-628808', 'mp-19017', 'mp-258', 'mp-1368']
    mpids = ['mp-1368']
    for mpid in mpids:
        new_struct = mpr.get_structure_by_material_id(mpid)
        starting_vol = new_struct.volume
        print 'Starting volume for {} = {}'.format(new_struct.composition, starting_vol)
        pv = VolumePredictor()
        # mp_data = pv.get_data(2, 0.05)
        # pv.fit(mp_data.structures, mp_data.volumes, mp_data.task_ids)
        # pv.save_avg_bondlengths("nelements_2_minbls.pkl")
        # pv.save_bondlengths("nelements_2_bls.pkl")
        pv.get_avg_bondlengths("nelements_2_minbls.pkl")
        a = pv.predict(new_struct)
        percent_volume_change = ((a.volume - starting_vol) / starting_vol) * 100
        print 'Predicted volume = {} with RMSE = {} and a volume change of {}%'.format(a.volume, a.rmse, percent_volume_change)
