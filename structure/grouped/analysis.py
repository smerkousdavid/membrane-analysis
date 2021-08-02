from typing import List, Tuple
import multiprocessing
import os
import posixpath
import queue
import threading

import cv2
import dask as da
import h5py as hf
import numpy as np
import pint
from structure import ureg, Q_
from structure.analysis.statistics import Statistics, compute_statistics
from structure.grouped.fpw import make_fpw_measurements
from structure.interactive.dataset import (DATA_PATH, LABEL_PATH, OUT_PATH,
                                           Dataset, create_input_dataset,
                                           create_input_output_dataset)

ANALYSIS_PATH = 'analysis'
COMPRESSION = 'lzf'
PIXEL_UNIT = Q_('pixel')
QP_ = lambda V: Q_(V, 'pixel')  # create a pixel quantity
COMPRESSION_OPTS = None


class GroupedAnalysis(object):
    def __init__(self, dataset: (str, Dataset), settings: dict = {}, scale_data: bool = True):
        self.ds = dataset if isinstance(dataset, Dataset) else Dataset(
            dataset, create_new=False)  # load an existing dataset from a file
        self.scaled = self.ds.is_scaled() and scale_data
        self.scaling = self.ds.get_dataset_scaling() if scale_data else None
        self.units = str((PIXEL_UNIT * self.scaling).units) if self.scaling is not None else str(PIXEL_UNIT.units)
        self.settings = settings
        self.threads = multiprocessing.cpu_count()

    def set_threads(self, num: int):
        self.threads = num

    def _make_dataset(self, group: hf.Group, name: str, data: (list, set, pint.Quantity, np.ndarray), dtype=np.double, cur_units: (str, pint.Quantity) = None, scale: bool = False) -> hf.Dataset:
        if name in group:
            del group[name]  # remove the current dataset

        # convert basic types
        if isinstance(data, (list, set)):
            data = np.array(data, dtype=dtype)

        # auto scale dataset
        if scale:
            # determine current unit type
            if cur_units is None:
                if isinstance(data, pint.Quantity):
                    cur_units = data.units
            cur_units, data = self._scale_data(
                cur_units if cur_units is not None else PIXEL_UNIT, data)  # rescale the data

        # auto-assume pixel units if no conversion units were specified
        if cur_units is None:
            cur_units = PIXEL_UNIT

        dset = group.create_dataset(name, data=(data if isinstance(
            data, np.ndarray) else data.magnitude), dtype=dtype, compression=COMPRESSION, compression_opts=COMPRESSION_OPTS)
        dset.attrs['units'] = str(cur_units) if isinstance(cur_units, pint.Unit) else str(Q_(cur_units).units)
        return dset

    def _scale_data(self, cur_units: (str, pint.Quantity), data: (np.ndarray, pint.Quantity)) -> tuple:
        if self.scaled and self.scaling is not None:
            # make sure there is data
            no_data = False
            if isinstance(data, np.ndarray):
                no_data = len(data) == 0
            else:
                no_data = len(data.magnitude) == 0

            # no point in scaling...
            if no_data:
                if cur_units is None and isinstance(data, pint.Quantity):
                    units = data.units
                elif cur_units is None:
                    units = PIXEL_UNIT
                else:
                    units = cur_units
                c_data = Q_(units) * self.scaling
            else:
                # attempt to rescale all the data
                c_data = (Q_(data, cur_units) if isinstance(
                    data, np.ndarray) else data) * self.scaling

            # make sure end scaling doesn't result in some weird pixel division
            if '/pixel' in str(c_data.units).replace(' ', ''):
                raise RuntimeError(
                    'Failed to scale data from ' + str(cur_units) + ' with ' + str(self.scaling) + ' resulting units ' + str(c_data.units) + 'proc' + str(Q_(data, cur_units)))

            # return new scaled data
            if no_data:
                return c_data.units, (data if isinstance(data, np.ndarray) else data.magnitude)
            return c_data.units, c_data.magnitude

        # nothing to do
        return cur_units, data

    def process_fpw(self, group_slits: hf.Group, group_membrane: hf.Group, data: np.ndarray):
        membrane = data[-2]  # second last = membrane edge layer
        slits = data[-1]  # last = slit layer

        # make the measurements
        settings = self.settings.get('fpw', {})
        results = make_fpw_measurements(
            membrane_layer=membrane,
            slit_layer=slits,
            draw=False,
            export=False,  # this is DISABLED for now as we're not using it
            settings=settings
        )

        # get the respective data
        data = results.get_data()

        # copy the data over to the output group
        # @TODO implement region references for each membrane group https://docs.h5py.org/en/stable/refs.html
        has_results = results.is_valid()
        all_points = []
        all_point_pairs = []
        all_arc_distances = []
        all_direct_distances = []
        all_membrane_ranges = []
        all_membrane_points = []
        all_membrane_distances = []

        # if not results.is_valid():
        #     cv2.imshow('mem', membrane* 255)
        #     cv2.imshow('slits', slits* 255)
        #     cv2.waitKey(0)

        if has_results:
            for mdata in data:
                all_points.append(mdata['points'])
                all_point_pairs.append(mdata['point_pairs'])
                all_arc_distances.append(mdata['arc_distances'])
                all_direct_distances.append(mdata['direct_distances'])
                all_membrane_ranges.append(mdata['membrane_ranges'])
                all_membrane_points.append(mdata['membrane_points'])
                all_membrane_distances.append(mdata['membrane_distance'])

            # combine the results and convert them to numpy arrays
            all_points = np.concatenate(all_points).astype(
                np.int32) if all_points else np.array([], np.int32)
            all_point_pairs = np.concatenate(all_point_pairs).astype(
                np.int32) if all_point_pairs else np.array([], np.int32)
            all_arc_distances = np.concatenate(all_arc_distances).astype(
                np.double) if all_arc_distances else np.array([], np.double)
            all_direct_distances = np.concatenate(all_direct_distances).astype(
                np.double) if all_direct_distances else np.array([], np.double)
            all_membrane_ranges = np.concatenate(all_membrane_ranges).astype(
                np.int32) if all_membrane_ranges else np.array([], np.int32)
            all_membrane_points = np.concatenate(all_membrane_points).astype(
                np.int32) if all_membrane_points else np.array([], np.int32)

        # create the h5 data
        group_slits.attrs['valid'] = has_results
        self._make_dataset(group_slits, 'slit_locs', all_points, dtype=np.int32)
        self._make_dataset(group_slits, 'slit_pairs', all_point_pairs, dtype=np.int32)
        self._make_dataset(group_slits, 'slit_arc_distances', all_arc_distances)
        self._make_dataset(group_slits, 'slit_direct_distances', all_direct_distances)
        self._make_dataset(group_slits, 'slit_membrane_ranges', all_membrane_ranges, dtype=np.int32)
        self._make_dataset(group_membrane, 'membrane_points', all_membrane_points, dtype=np.int32)
        self._make_dataset(group_membrane, 'membrane_distances', np.array(all_membrane_distances).astype(np.double))

    def process_single(self, output: hf.Group, name: str):
        # get the data
        data = self.ds.get(name, None)
        if data is None:
            raise RuntimeError(
                'Data located at the name of ' + name + ' has no data in it')

        # load the data into memory as a numpy array
        np_data = data[:]

        # create the results group
        rep_name = self.__fix_out_to_analysis(name)
        # set the output segmented image to have this analysis name
        data.attrs['analysis'] = rep_name
        # set the input image to have this analysis name
        self.ds.get(data.attrs['input']).attrs['analysis'] = rep_name

        # create the new group for the current file and add relevant attributes
        base_group = output.require_group(rep_name)
        base_group.attrs['output'] = data.name  # output dataset/group path
        # input dataset/group path
        base_group.attrs['input'] = data.attrs['input']
        slit_group = base_group.require_group('slits')
        edge_group = base_group.require_group('membrane_edges')

        # add the processed image data
        self.process_fpw(slit_group, edge_group, np_data)

    def __analysis_process_consumer(self, _input: queue.Queue):
        while True:
            item = _input.get()

            # if None let's exit the thread
            if item is None:
                break

            # process the single data
            self.process_single(item['output'], item['out_data'])

            _input.task_done()

    def run(self, blocking: bool = True, force: bool = False):
        # let's determine if the dataset is dirty or not (as in file changes have been made) to continue
        if not self.ds.is_dirty() and not force and ANALYSIS_PATH in self.ds.data:
            return []  # everything has already been processed

        # get all input files
        input_names = self.ds.get_np('labels/input')

        # create the output groups
        output = self.ds.require_group(ANALYSIS_PATH)

        # start the threads
        threads = []
        in_queue = queue.Queue()
        for _ in range(self.threads):
            thread = threading.Thread(
                target=self.__analysis_process_consumer, args=(in_queue,))
            thread.start()
            threads.append(thread)

        # process each input file
        unprocessed = []
        for name in input_names:
            name = name.decode('utf-8')  # convert to string

            # get the input object
            in_data = self.ds.get(name)

            # get the path to the output data (usually the segmented data)
            out_data = in_data.attrs.get('output', None)

            # make sure there is output data
            if out_data is None:
                unprocessed.append(name)
                continue
            elif len(out_data) == 0:
                unprocessed.append(name)
                continue

            # push the queue to process this file
            in_queue.put({
                'output': output,
                'out_data': out_data
            })

        # kill the remaining threads (end of Queue)
        for i in range(self.threads * 2):
            in_queue.put(None)

        # if blocking, then wait for all of them to finish
        if blocking:
            for thread in threads:
                thread.join()

        # let's mark the file as no longer being dirty
        self.ds.set_dirty(False)

        return unprocessed

    def get_data_from_group(self, group_or_data: (str, bytes, hf.Group, hf.Dataset) = None, name: str = 'slit_arc_distances', with_dask: bool = False, with_numpy: bool = True):
        res = self.ds.get_all_ds_recursive(group_or_data, name=name)

        if with_dask:
            return self.data_to_dask(res)
        elif with_numpy:
            return self.data_to_np(res)
        return res

    def __fix_out_to_analysis(self, path: (str, bytes, hf.Group)):
        if isinstance(path, bytes):
            path = path.decode('utf-8')
        elif isinstance(path, hf.Group):
            path = path.name
        if path.endswith('/' + OUT_PATH) or path.endswith('/' + DATA_PATH):
            path += '/'  # add ending to be replaced
        return path.replace('/' + OUT_PATH + '/', '/' + ANALYSIS_PATH + '/').replace('/' + DATA_PATH + '/', '/' + ANALYSIS_PATH + '/')

    def data_to_dask(self, data: list, axis: int = 0):
        return da.concatenate([da.from_array(d) for d in data], axis=axis)

    def data_to_np(self, data: list, axis: int = 0):
        if len(data) == 0:
            return np.array(data, dtype=np.double)
        return np.concatenate(data, axis=axis)

    def get_stats_on_data(self, group_or_data: (str, bytes, hf.Group, hf.Dataset) = None, name: str = 'slit_arc_distances') -> Statistics:
        data = self.get_data_from_group(self.__fix_out_to_analysis(
            group_or_data), name, with_dask=False, with_numpy=True).astype(np.double)
        stats = compute_statistics(data)
        return stats

    def get_all_stats(self, file_name: str, group: (str, bytes, hf.Group, ) = None) -> dict:
        return {
            'file': file_name,
            'units': self.units,
            'group': group if isinstance(group, (str, bytes)) else group.name,
            'slit_arc_distances': self.get_stats_on_data(group, 'slit_arc_distances'),
            'slit_direct_distances': self.get_stats_on_data(group, 'slit_direct_distances')
        }

    def get_stats_at_depth(self, depth: int = 0, stats_type: str='groups'):
        """ Scans all subgroups to get their collective stats """
        path = posixpath.join(LABEL_PATH, 'depth_out_' + stats_type)
        if str(depth) not in self.ds.get(path):
            raise RuntimeError('The depth ' + str(depth) + ' does not exist')

        # convert groups to their respective analysis equivalents
        groups = [self.__fix_out_to_analysis(d) for d in self.ds.get_str(
            posixpath.join(path, str(depth)))]

        # get the stats for each group
        return [self.get_all_stats(os.path.basename(group), group) for group in groups]

    def get_stats_at_depths(self, depths: list, stats_type: str='groups'):
        """ Scans multiple sub-groups and gets all of their stats """
        return [{'depth': int(d), 'stats': self.get_stats_at_depth(int(d), stats_type=stats_type)} for d in depths]

# def analyze_fpw(data: np.ndarray) -> dict:

#     in_image = cv2.imread(input, cv2.IMREAD_GRAYSCALE)
#     layers = tifffile.imread(output)

#     layers = np.ascontiguousarray(layers)
#     layers_uint = (layers > 50).astype(np.uint8)
#     membrane = layers_uint[-2]
#     slits = layers_uint[-1]

#     # make the measurements
#     fpw = make_fpw_measurements(membrane, slits, draw=False, export=True)

#     back_layer = ResultLayer('background', 'Background')
#     back_layer.draw_image(in_image, in_image.shape[::-1])

#     exports = [back_layer]
#     if fpw.is_valid():
#         exports += fpw.get_export()

#     basic = os.path.splitext(os.path.basename(input))[0]
#     exp.write_export(os.path.join(SAVE_OUTPUT, basic + '.html'), os.path.basename(input), exports)
#     return fpw


def test():
    dset = create_input_output_dataset('C:\\Users\\smerk\\Documents\\Ground Truth Project\\Fabry - 09-0598\\09-0598 blk 1-1\\surs',
                                       'C:\\Users\\smerk\\Documents\\Ground Truth Project\\Fabry - 09-0598\\09-0598 blk 1-1\\out_class', new_file=False, keep_existing=True,
                                       scaling_data={
                                           'ground truth project\\fabry - 09-0598\\09-0598 blk 1-1\\surs\\09--_4339.tif': '10 nm/pixel',
                                       })
    # dset = Dataset('dataset.h5', create_new=False)
    sset = dset.save_to_scaled_dataset()
    analysis = GroupedAnalysis(sset)
    unprocessed = analysis.run()
    # print(dset.get_all_ds_recursive(start='/analysis', name='slit_arc_distances'))
    stats = analysis.get_stats_at_depth(0)
    dset.close()
    print(stats)
    # print('WARNING! The following files were not processed', unprocessed)

    # print(dset.get_all_ds_recursive(attribute='depth', attribute_equals=2))


if __name__ == '__main__':
    print('starting')
    # import cProfile
    # cProfile.run('test()', 'profile.prof')
    test()
    print('done')
