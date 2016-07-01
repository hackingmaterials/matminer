import os
import re
from pymatgen import MPRester, Element, Composition, Structure
from collections import defaultdict, namedtuple
from pymatgen.analysis.structure_analyzer import VoronoiCoordFinder
from sklearn.metrics import mean_squared_error
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from matminer.descriptors.composition_features import get_pymatgen_eldata_lst, get_std
import plotly.plotly as py

mpr = MPRester()

module_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
bl_dir = os.path.join(module_dir, 'data_bondlengths')
data_dir = os.path.join(module_dir, 'data')

data = namedtuple('data', 'task_ids structures volumes')
struct_data = namedtuple('struct_data', 'min_bond_length task_id')


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
        Get minimum bond lengths in a structure by bond type.

        :param structure: pymatgen structure object
        :return: (dict) with bond types as keys in the format 'A-B', and values as namedtuples that contain the
        minimum bond distance and number of bonds
        """
        bondlengths = defaultdict(list)
        min_bondlengths = {}
        minbond_details = namedtuple('minbond_details', 'min_dist no_of_bonds')
        for site_idx, site in enumerate(structure.sites):
            site_element_str = ''.join(re.findall(r'[A-Za-z]', site.species_string))
            try:
                voronoi_sites = VoronoiCoordFinder(structure).get_coordinated_sites(site_idx, tol=0.5)
            except RuntimeError as r:
                print 'Error for site {} in {}: {}'.format(site, structure.composition, r)
                continue
            except ValueError as v:
                print 'Error for site {} in {}: {}'.format(site, structure.composition, v)
                continue
            print 'Main {}'.format(site)
            for vsite in voronoi_sites:
                print 'Neighbor {}'.format(vsite)
                s_dist = np.linalg.norm(vsite.coords - site.coords)
                # TODO: avoid "continue" statements if possible-it is dead simply to write this code without "continue"
                if s_dist < 0.1:
                    continue
                vsite_element_str = ''.join(re.findall(r'[A-Za-z]', vsite.species_string))
                bond = '-'.join(sorted([site_element_str, vsite_element_str]))
                bondlengths[bond].append(s_dist)
        for bondtype in bondlengths:
            min_length = bondlengths[bondtype][0]
            for length in bondlengths[bondtype]:
                if length < min_length:
                    min_length = length
            min_bondlengths[bondtype] = minbond_details(min_dist=min_length, no_of_bonds=len(bondlengths[bondtype]))
        return min_bondlengths

    def fit(self, structures, volumes, ids=None):
        """
        Given a set of input structures, it stores minimum bond lengths (one minimum for each structure)
        in a defaultdict(list), and average of the minimum bond lengths in a dictionary by bond type (keys).

        :param structures: (list) list of pymatgen structure objects
        :param volumes: (list) corresponding list of volumes of structures
        :param ids: (list) corresponding list of MP task_ids. Default=None
        """
        for idx, struct in enumerate(structures):
            struct_minbls = self.get_bondlengths(struct)
            for bond in struct_minbls:
                if ids is not None:      # TODO: add else statement
                    self.bond_lengths[bond].append(struct_data(min_bond_length=struct_minbls[bond].min_dist,
                                                               task_id=ids[idx]))
        for bond in self.bond_lengths:
            total = 0
            for item in self.bond_lengths[bond]:
                total += item.min_bond_length
            self.avg_bondlengths[bond] = total / len(self.bond_lengths[bond])

    def get_rmse(self, struct_minbls, volume_factor):
        """
        Calculates root mean square error between minimum bond lengths in a structure and
        the average expected bond lengths.

        :param struct_minbls:(dict) of minimum bond lengths as output by the function "get_bondlengths()"
        :param volume_factor:(float) percentage change in volume. Eg: 0.81 represents 81% of starting volume.
        :return: (float) root mean square error
        """
        y = []
        y_avg = []
        for bond in struct_minbls:
            new_min_bondlength = struct_minbls[bond].min_dist * (volume_factor ** (1.0 / 3.0))
            y.extend([new_min_bondlength] * struct_minbls[bond].no_of_bonds)
            if bond in self.avg_bondlengths:
                y_avg.extend([self.avg_bondlengths[bond]] * struct_minbls[bond].no_of_bonds)
            else:
                print 'Warning missing bond length for {}'.format(bond)
                el1, el2 = bond.split("-")
                r1 = float(Element(el1).atomic_radius)
                r2 = float(Element(el2).atomic_radius)
                y_avg.extend([(r1+r2)] * struct_minbls[bond].no_of_bonds)
        return mean_squared_error(y, y_avg) ** 0.5

    def predict(self, structure):
        """
        Predict volume of a given structure based on rmse against average bond lengths from a set of input structures.
        Note: run this function after initializing the "self.avg_bondlengths" variable, through either of the functions
        fit() or get_avg_bondlengths().

        :param structure: (structure) pymatgen structure object to predict the volume of
        :return: (namedtuple) predicted volume (float), structure (pymatgen structure object) and its rmse (float)
        """
        prediction_results = namedtuple('prediction_results', 'volume structure rmse')
        starting_volume = structure.volume
        predicted_volume = starting_volume
        min_rmse = self.get_rmse(self.get_bondlengths(structure), 1)  # Get starting RMSE for original structure
        starting_minbls = self.get_bondlengths(structure)
        min_percentage = 1000
        for i in range(800, 1201):
            vol_factor = (i * 0.001)
            test_volume = vol_factor * starting_volume
            structure.scale_lattice(test_volume)
            test_rmse = self.get_rmse(starting_minbls, vol_factor)
            if test_rmse < min_rmse:
                min_rmse = test_rmse
                predicted_volume = test_volume
                min_percentage = i
        if min_percentage < 850 or min_percentage > 1150:
            return self.predict(structure)
        else:
            return prediction_results(volume=predicted_volume, structure=structure, rmse=min_rmse)

    def save_avg_bondlengths(self, filename):
        """
        Save the class variable "self.avg_bondlengths" calculated by fit().

        :param filename: name of file to store to
        """
        with open(os.path.join(bl_dir, filename), 'w') as f:
            pickle.dump(self.avg_bondlengths, f, pickle.HIGHEST_PROTOCOL)

    def save_bondlengths(self, filename):
        """
        Save the class variable "self.bond_lengths" calculated by fit().

        :param filename: name of file to store to
        """
        with open(os.path.join(bl_dir, filename), 'w') as f:
            pickle.dump(self.bond_lengths, f, pickle.HIGHEST_PROTOCOL)

    def get_avg_bondlengths(self, filename):
        """
        Extract the saved average bond lengths and save them in the class variable "self.avg_bondlengths".

        :param filename: name of file to extract average bond lengths from
        :return:
        """
        with open(os.path.join(bl_dir, filename), 'r') as f:
            self.avg_bondlengths = pickle.load(f)


def save_predictions(structure_data):
    """
    Save results of the predict() function in a dataframe.

    :param structure_data: (namedtuple) of lists of task_ids, structures, and volumes, as output by get_data()
    :return:
    """
    df = pd.DataFrame()
    cv = VolumePredictor()
    cv.get_avg_bondlengths("nelements_2_minbls.pkl")
    x = 0
    y = 0
    for idx, structure in enumerate(structure_data.structures):
        x += 1
        if x % 10 == 0:
            print 'Current completed = {}'.format(x)
        try:
            b = cv.predict(structure)
        except ValueError as e:
            print 'ValueError for {}: {}'.format(structure_data.task_ids[idx], e)
            y += 1
            print 'INCOMPLETE = {}'.format(y)
            continue
        df = df.append({'task_id': structure_data.task_ids[idx],
                        'reduced_cell_formula': Composition(structure.composition).reduced_formula,
                        'starting_volume': structure_data.volumes[idx], 'predicted_volume': b.volume,
                        'percentage_volume_change': ((b.volume - structure_data.volumes[idx])/structure_data.volumes[idx]) * 100},
                       ignore_index=True)
        print 'Done for {}'.format(structure_data.task_ids[idx])
    df.to_pickle(os.path.join(data_dir, 'cv_1_on_2.pkl'))


class VolumePredictorSimple:
    def __init__(self, cutoff=4, ionic_factor=0.125):
        """
        Predicts volume; given a structure, finds the minimum volume such that
        no two sites are closer than sum of their atomic radii. The sum of
        atomic radii is modified for ionicity using an ionic_factor that
        depends on electronegativity difference.
        Args:
            cutoff: (float) cutoff for site pairs (added to site radius)
                in Angstrom. Increase if your initial structure guess
                is extremely bad (atoms way too far apart). In all other cases,
                increasing cutoff gives same answer but at lower performance.
        """
        self.cutoff = cutoff
        if ionic_factor > 0.25:
            raise ValueError("specified ionic factor is out of range!")
        self.ionic_factor = ionic_factor

    def predict(self, structure):
        """
        Given a structure, returns back a volume
        Args:
            structure: (Structure)
        Returns:
            (float) expected volume of structure
        """

        if not structure.is_ordered:
            raise ValueError("VolumePredictorSimple requires "
                             "ordered structures!")

        smallest_distance = None
        ionic_mix = get_std(get_pymatgen_eldata_lst(structure.composition, 'X')) * 0.25
        # ionic_mix = abs(el1.X - el2.X) * self.ionic_factor

        for site in structure:
            el1 = site.specie
            if el1.atomic_radius:
                x = structure.get_neighbors(site,
                                            el1.atomic_radius + self.cutoff)

                for site2, dist in x:
                    el2 = site2.specie
                    if el2.atomic_radius:
                        r1 = el1.average_ionic_radius * ionic_mix +\
                             el1.atomic_radius * (1-ionic_mix) if \
                            el1.average_ionic_radius else el1.atomic_radius

                        r2 = el2.average_ionic_radius * ionic_mix +\
                             el2.atomic_radius * (1-ionic_mix) if \
                            el2.average_ionic_radius else el2.atomic_radius

                        expected_dist = float(r1 + r2)

                        if not smallest_distance or dist/expected_dist \
                                < smallest_distance:
                            smallest_distance = dist/expected_dist
                    else:
                        warnings.warn("VolumePredictor: no atomic radius data for {}".
                              format(el2))

            else:
                warnings.warn("VolumePredictor: no atomic radius data for {}".
                              format(el1))

        if not smallest_distance:
            raise ValueError("Could not find any bonds in this material!")

        return structure.volume * (1/smallest_distance)**3


def plot_plotly(dataframe):
    fig = {
        'data': [
            {
                'x': dataframe['starting_volume'],
                'y': dataframe['predicted_volume'],
                'text': dataframe['reduced_cell_formula'],
                'mode': 'markers',
                'name': '0.2*std(X_comp)'},
        ],
        'layout': {
            'xaxis': {'title': "Expected volume"},
            'yaxis': {'title': "Predicted volume"}
        }
    }

    # IPython notebook
    # py.iplot(fig, filename='aj_2')

    url = py.plot(fig, filename='aj_02')


if __name__ == '__main__':
    pd.set_option('display.width', 1000)
    # pv = VolumePredictor()
    # mp_data = pv.get_data(2, 0.05)
    # pv.fit(mp_data.structures, mp_data.volumes, mp_data.task_ids)
    # pv.save_avg_bondlengths("nelements_2_minbls.pkl")
    # pv.save_bondlengths("nelements_2_bls.pkl")
    '''
    # mpids = ['mp-628808', 'mp-19017', 'mp-258', 'mp-1368']
    # mpids = ['mp-258', 'mp-23210', 'mp-1138', 'mp-149', 'mp-22914']
    mpids = ['mvc-11598']
    for mpid in mpids:
        new_struct = mpr.get_structure_by_material_id(mpid)
        starting_vol = new_struct.volume
        print 'Starting volume for {} = {}'.format(new_struct.composition, starting_vol)
        pv = VolumePredictor()
        pv.get_avg_bondlengths("nelements_2_minbls.pkl")
        try:
            a = pv.predict(new_struct)
        except ValueError:
            continue
        percent_volume_change = ((a.volume - starting_vol) / starting_vol) * 100
        print 'Predicted volume = {}, with RMSE = {} and a volume change of {}%'.format(a.volume,
                                                                    a.rmse, percent_volume_change)
    '''
    with open(os.path.join(data_dir, "mp_rawdata_2.pkl"), 'r') as f:
        mp_data = pickle.load(f)
    # save_predictions(mp_data)
    # '''
    # df = pd.read_pickle(os.path.join(data_dir, 'aj_2_old1.pkl'))
    # print df
    # print df.sort(['percentage_volume_change'])
    # a = [x for x in df['percentage_volume_change'].tolist()]
    # print(sum(a)/len(a))
    # df.plot(x='starting_volume', y='predicted_volume', kind='scatter')
    # plt.plot([0,8000], [0,8000], 'k-')
    # df.plot(x='starting_volume', y='percentage_volume_change', kind='scatter')
    # plt.show()
    # plot_plotly(df)
    # '''
    x = 0
    df = pd.DataFrame()
    for idx, structure in enumerate(mp_data.structures):
        x += 1
        if x % 10 == 0:
            print 'Current completed = {}'.format(x)
        try:
            new_vol = VolumePredictorSimple().predict(structure)
        except ValueError as e:
            print e
            continue
        df = df.append({'task_id': mp_data.task_ids[idx],
                        'reduced_cell_formula': Composition(structure.composition).reduced_formula,
                        'starting_volume': mp_data.volumes[idx], 'predicted_volume': new_vol,
                        'percentage_volume_change': ((new_vol - mp_data.volumes[idx])/mp_data.volumes[idx]) * 100},
                       ignore_index=True)
        print 'Done for {}'.format(mp_data.task_ids[idx])
    df.to_pickle(os.path.join(data_dir, 'aj_025.pkl'))
    print df.sort(['percentage_volume_change'])
    # '''
    # '''
    # s = mpr.get_structure_by_material_id('mp-24972')
    # print s.volume
    # new_volume = VolumePredictorSimple().predict(s)
    # print new_volume




