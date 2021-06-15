import posixpath
import h5py as hf
import numpy as np
import os
import threading
import multiprocessing
import tifffile
import cv2
import queue
import json
import re

# string datatype for h5 files
HF_STR = hf.string_dtype(encoding='utf-8')
LABEL_PATH = 'labels'
DATA_PATH = 'data'
OUT_PATH = 'out'

# used for data compression
COMPRESSION = 'lzf'
COMPRESSION_OPTS = None  # no options for lzf


def get_file_name(path: str):
    if path is None:
        return None
    return os.path.splitext(os.path.basename(str(path)))[0]


def get_file_ext(path: str):
    if path is None:
        return None
    return os.path.splitext(os.path.basename(str(path)))[1].replace('.', '').lower().strip()


class Dataset(object):
    def __init__(self, file: str, create_new: bool = False):
        if not os.path.isfile(file):
            create_new = True  # we have to write a new h5 file
        self.data = hf.File(file, 'w' if create_new else 'a')
        self.data_loader = {}
        self.data_labels = {}
        self.callbacks = {}
        self.keep_existing = False
        self.ignore_extensions = set()
        self.threads = multiprocessing.cpu_count()

        # populate the default image loader
        self.set_custom_data_loader(cv2.imread, ['png', 'jpg', 'jpeg', 'bmp'])
        self.set_custom_data_loader(tifffile.imread, ['tif', 'tiff'])

    def set_threads(self, num: int):
        self.threads = num

    def set_keep_existing(self, existing: bool):
        self.keep_existing = existing

    def _make_dict_struct(self, group: hf.Group, data: (tuple, list, set, dict), tkey: str = None, add_key: str = ''):
        if data is None:
            return  # nothing to do

        # iter through dict and do different things depending on the struct
        is_dict = isinstance(data, dict)
        keys = data.keys() if is_dict else data

        # make sure tkey is defined
        if tkey is None and not is_dict:
            raise ValueError(
                'tkey must be defined if data structure is not a dictionary')

        for key in keys:
            if is_dict:
                val = data[key]
            else:
                val = key  # we're working with a non-dict structure
                key = tkey

            # add to key
            key += str(add_key)

            # determine what to do
            if val is None:
                group.create_dataset(key, dtype='f')
            elif isinstance(val, dict):
                ngroup = group.create_group(key)
                self._make_dict_struct(ngroup, val)
            elif isinstance(val, (tuple, list, set)):
                make_group = True
                if len(val) > 0:
                    if not isinstance(val[0], (tuple, list, set, dict)):
                        group.create_dataset(key, data=val)
                        make_group = False

                if make_group:
                    ngroup = group.create_group(key)
                    ngroup.attrs['list'] = True
                    ngroup.attrs['size'] = len(val)

                    if len(val) > 0:
                        for ind, item in enumerate(val):
                            sub_group = ngroup.create_group(str(ind))
                            self._make_dict_struct(sub_group, item, tkey=key)
            else:
                group.create_dataset(key, data=val.encode(
                    'utf-8') if isinstance(val, str) else val, dtype=HF_STR if isinstance(val, str) else None)

    def set_callback(self, name: str, function):
        self.callbacks[name] = function

    def set_ignore_extensions(self, exts: list):
        self.ignore_extensions = set(
            [str(ext).replace('.', '').lower() for ext in exts])

    def json_loader(self, group: hf.Group, path: str) -> np.ndarray:
        if path is None or group is None:
            raise ValueError('path nor group must not be None')

        with open(path, 'r') as jdata:
            data = jdata.read()
            if len(data) == 0 or (not data.startswith('{')) or (not data.endswith('}')):
                raise ValueError('json data loaded from ' +
                                 path + ' is not a valid json object')
            data = json.loads(data)

        # now convert the tree like structure into h5 format
        name = get_file_name(path)
        ext = get_file_ext(path)

        # new group for file
        ngroup = group.create_group(name)
        ngroup.attrs['file_extension'] = ext

        # load the data
        self._make_dict_struct(ngroup, data)

    def _clean_ext(self, ext: str) -> str:
        if ext is None:
            raise ValueError('extensions cannot be none')
        ext = ext.replace('.', '').lower().strip()
        if len(ext) == 0:
            raise ValueError(
                'after cleaning extension file was invalid ' + ext)
        return ext

    def set_custom_data_loader(self, loader, extensions):
        failed = False
        if extensions is None or loader is None:
            failed = True
        elif len(extensions) == 0:
            failed = True

        if failed:
            raise ValueError('Data loader and extensions cannot be empty')

        # add all of the loaders for the specified extensions
        for ext in extensions:
            ext = self._clean_ext(ext)
            self.data_loader[ext] = loader

    def set_custom_data_labeler(self, regex: str, labels: list = None):
        if regex is None:
            raise ValueError('Regex label matching cannot be None')

        # set labeling
        self.data_labels[regex] = labels

    def _get_file_labels_from_path(self, path: str) -> list:
        # try to match all of the files
        f_name = get_file_name(path)
        f_ext = get_file_ext(path)
        # a normalized full file name
        name = os.path.join(os.path.dirname(
            path).lower(), f_name + '.' + f_ext)

        # attempt to find a corresponding label
        labels = []
        for key in self.data_labels.keys():
            if re.match(key, name) is not None:
                labels.extend(self.data_labels[key])
        return hf.Empty('S') if len(labels) == 0 else np.array(labels, dtype='S')

    def make_dataset_from_file(self, path: str, group: (hf.File, hf.Group, str) = None, cast_dtype: str = None, metadata: dict = None, custom_name: str = None, add_labels: list = None, label_group: (hf.Group) = None, call_make_file_callback: bool = True, depth: int = None, **kwargs) -> list:
        """ Loads a file as a numpy array with a casting type if specified """

        if group is None or isinstance(group, Dataset):
            group = self.data

        name = get_file_name(path) if custom_name is None else custom_name
        ext = get_file_ext(path)
        if ext is None:
            raise ValueError('the file ' + path +
                             ' does not have any extension')
        elif len(ext) == 0:
            raise ValueError('the file ' + path +
                             ' does not have any extension')

        # load the specified file
        labels = hf.Empty('S')
        name_path = None

        if ext.replace('.', '').lower() in self.ignore_extensions:
            raise NotImplementedError(
                'The following file in in the ignored file list ' + path)

        if ext in self.data_loader:
            if self.keep_existing and ((group in self.data) if isinstance(group, str) else (name in group)):
                return hf.Empty('S'), None

            data = np.asarray(self.data_loader[ext](path))

            if cast_dtype is not None:
                data = data.astype(cast_dtype)

            # create the full path of the string
            if isinstance(group, str):
                dset = self.data.create_dataset(
                    group, data=data, dtype=cast_dtype, compression=COMPRESSION, compression_opts=COMPRESSION_OPTS, **kwargs)
            else:
                dset = group.create_dataset(
                    name, data=data, dtype=cast_dtype, compression=COMPRESSION, compression_opts=COMPRESSION_OPTS, **kwargs)
            dset.attrs['file_extension'] = ext
            name_path = dset.name
            labels = self._get_file_labels_from_path(path)
            dset.attrs['labels'] = labels

            # set the files depth attribute
            if depth is not None:
                dset.attrs['depth'] = np.int32(depth)

            if 'make_file' in self.callbacks and call_make_file_callback:
                self.callbacks['make_file']({
                    'name_path': name_path,
                    'labels': [] if (isinstance(labels, hf.Empty) or labels is None) else [i.decode('utf-8') for i in labels],
                    'group': group,
                    'extension': ext,
                    'dataset': self
                })

            # copy metadata
            if metadata is not None:
                for key in metadata.keys():
                    dset.attrs[key] = metadata[key]

        elif 'json' == ext or 'js' == ext:
            self.json_loader(group, path)
        else:
            raise NotImplementedError('Could not load data for the extension type ' +
                                      ext + ' please make sure the custom data loader is specified')

        # add a label when creating this file
        if name_path is not None and add_labels is not None and label_group is not None:
            for label in add_labels:
                self.add_label_list(
                    label, [name_path], label_group=label_group)

        return labels, name_path

    def add_label_list(self, label: (bytes, str), data: (list, set, tuple, np.ndarray), label_group: (hf.File, hf.Group)):
        if isinstance(label, bytes):  # convert from utf-8 encoded bytes to string
            label = label.decode('utf-8')

        data = np.array(data, dtype='S')
        if data.shape:
            if data.shape[0] > 0:  # if there are elements
                size = max(int(np.amax(np.char.str_len(data))), len(label))
            else:
                size = len(label)
        else:
            size = len(label)

        # create new datatype for dataset (to handle the new max fixed size string)
        dtype = hf.string_dtype('utf-8', size)

        if label in label_group:
            cur_data = label_group.get(label, None)

            # this is an update so get new largest size
            if cur_data is not None and not isinstance(cur_data, hf.Empty):
                size = max(int(np.amax(np.char.str_len(cur_data))), size)

                # create new datatype for dataset
                dtype = hf.string_dtype('utf-8', size)

            del label_group[label]  # remove from the file
            label_group.create_dataset(label, data=np.append(
                cur_data, data).astype(dtype), dtype=dtype)  # append the new labels
        else:
            label_group.create_dataset(label, data=data.astype(
                dtype), dtype=dtype)  # create the new data

    def get_all_ds_recursive(self, start: (hf.File, hf.Group, bytes, str) = None, attribute: (str, bytes) = None, attribute_equals: (str, int, bytes) = None, name: (str, bytes) = None):
        if start is None:
            start = self.data
        elif isinstance(start, (bytes, str)):
            start = self.data[start]

        # let's start scanning recursively
        results = []
        for key in start.keys():
            val = start[key]
            if isinstance(val, hf.Group):
                results.extend(self.get_all_ds_recursive(
                    start=val, attribute=attribute, attribute_equals=attribute_equals, name=name))
            elif isinstance(val, hf.Dataset):
                passed = True
                if attribute is not None:
                    if attribute not in val.attrs:
                        passed = False
                    elif val.attrs[attribute] != attribute_equals:
                        passed = False

                if name is not None:
                    if os.path.basename(val.name) != name:
                        passed = False

                if passed:
                    results.append(val)
        return results

    def __labeled_dataset_file_consumer(self, queue: queue.Queue, output: queue.Queue):
        while True:
            item = queue.get()

            # quit the worker
            if item is None:
                break

            try:
                labels, name_path = self.make_dataset_from_file(
                    item['full'], item['group'], depth=item['depth'])

                if labels is not None and name_path is not None:
                    # array of labels not an empty dataset
                    if not isinstance(labels, hf.Empty):
                        for label in labels:
                            label_ucode = label.decode('utf-8')
                            output.put(
                                (label_ucode, name_path.encode('utf-8'), item['depth']))
                            # if label_ucode in tracked_labels:
                            #     tracked_labels[label_ucode].append(name_path.encode('utf-8'))
                            # else:
                            #     tracked_labels[label_ucode] = [name_path]
            except NotImplementedError as err:
                # output an error (possibly okay)
                output.put({'error': err, 'fatal': False})
            except Exception as err:
                # always fatal so let's quit
                output.put({'error': err, 'fatal': True})
                break

            queue.task_done()

    def make_dataset_from_folder(self, path: str, group: (hf.File, hf.Group) = None, recursive: bool = True, ignore_invalid: bool = True, label_group: (hf.File, hf.Group) = None, tracked_labels: dict = None, depth_groups: dict = None, depth_labels: dict = None, depth: int = 0, _worker_queues: dict = None):
        if path is None or group is None:
            raise ValueError('Neither path nor group can be None')

        if not os.path.isdir(path):
            raise RuntimeError('The path ' + path +
                               ' is not a folder. Cannot load dataset')

        # make ref
        if group is None or isinstance(group, Dataset):
            group = self.data

        # walk the directories and add the relative paths
        if tracked_labels is None:
            original = True
            tracked_labels = {}
            depth_labels = {
                # starting with a depth of 0 (which are in the root folder)
                '0': [],
                '1': []
            }
            depth_groups = {
                '0': [group.name],  # let the root folder always be depth 0
                '1': []
            }
            _worker_queues = {
                'file': queue.Queue(),
                'out': queue.Queue()
            }

            # start the consumer threads
            all_threads = []
            for ind in range(self.threads):
                thread = threading.Thread(target=self.__labeled_dataset_file_consumer, args=(
                    _worker_queues['file'], _worker_queues['out']), name='Dataset Consumer Thread %d' % ind)
                thread.start()
                all_threads.append(thread)
        else:
            original = False

        for item in os.listdir(path):
            full = os.path.join(path, item)
            if os.path.isdir(full):
                ngroup = group.require_group(item)
                ngroup.attrs['depth'] = np.int32(depth)

                # add this folder to the specified depth (new sub-folder)
                ndepth = depth + 1
                sndepth = str(ndepth)
                if sndepth not in depth_groups:
                    depth_groups[sndepth] = []
                if sndepth not in depth_labels:
                    depth_labels[sndepth] = []
                depth_groups[sndepth].append(ngroup.name)

                # handle group making callbacks
                if 'make_group' in self.callbacks:
                    self.callbacks['make_group']({
                        'name_path': ngroup.name,
                        'group': group,
                        'depth': depth
                    })

                # keep scanning
                if recursive:
                    self.make_dataset_from_folder(full, ngroup, label_group=label_group, tracked_labels=tracked_labels,
                                                  depth=ndepth, depth_groups=depth_groups, depth_labels=depth_labels, _worker_queues=_worker_queues)
            else:
                # tell the consumer threads to process this file
                _worker_queues['file'].put({
                    'full': full,
                    'depth': depth,
                    'group': group
                })

        if original:
            # let's fill the queues with Nones to quit the extra threads
            # just to be safe (it takes nearly no time) let's overfill the queues
            for i in range(self.threads * 2):
                _worker_queues['file'].put(None)

            # let's wait for all threads to finish
            for thread in all_threads:
                thread.join()

        # construct a set of all the used labels so we can group them together (we'll do this at the end so in the original recursive layer)
        if label_group is not None and original and tracked_labels is not None:
            # pull all of the items from the output queue
            while not _worker_queues['out'].empty():
                item = _worker_queues['out'].get()
                if isinstance(item, dict) and ((not ignore_invalid) or item['fatal']):
                    raise item['error']
                label, value, sdepth = item  # decompose otherwise

                # add the file label
                if label in tracked_labels:
                    tracked_labels[label].append(value)
                else:
                    tracked_labels[label] = [value]

                # add this file to the specified depth
                depth_labels[str(sdepth)].append(value)

                _worker_queues['out'].task_done()

            all_labels = set()
            for val in self.data_labels.values():
                if val:
                    for item in val:
                        all_labels.add(item)

            # add them to the h5 file grouped together
            for label in all_labels:
                if label in tracked_labels:
                    self.add_label_list(
                        label, tracked_labels[label], label_group=label_group)

            # let's add the depth labels to the label_group
            depth_fgroups = label_group.require_group('depth_files')
            depth_ggroups = label_group.require_group('depth_groups')
            for depth in depth_groups.keys():
                # add all files at this depth
                self.add_label_list(
                    depth, depth_labels[depth], label_group=depth_fgroups)

                # add all groups at this depth
                self.add_label_list(
                    depth, depth_groups[depth], label_group=depth_ggroups)

    def get(self, path: str, default=None) -> hf.Dataset:
        return self.data.get(path, default)

    def get_np(self, path: str, default=None) -> np.ndarray:
        return self.data.get(path, default)[:]

    def get_str(self, path: str, default=None) -> np.ndarray:
        return self.data.get(path, default).asstr()[:]

    def make_group(self, name: str, parent=None, **kwargs):
        if parent is None:
            parent = self.data
        return parent.create_group(name, **kwargs)

    def require_group(self, name: str, parent=None, **kwargs):
        if parent is None:
            parent = self.data
        return parent.require_group(name, **kwargs)

    def close(self):
        self.data.close()


def create_input_dataset(on_folder: str, file_path: str = 'dataset.h5', root_group: str = 'data', input_label='input',
            ignore_exts: list = ['json', 'txt', 'js', 'pdf'], callbacks: dict = {}, new_file: bool=True, keep_existing: bool=False) -> Dataset:
    if on_folder is None:
        raise ValueError('The folder cannot be None')

    if not os.path.isdir(on_folder):
        raise RuntimeError('The folder ' + on_folder + ' does not exist')

    # create the new default dataset
    dset = Dataset(file_path, create_new=new_file)
    dset.set_keep_existing(keep_existing)
    dset.set_custom_data_labeler(r'.*\.(tiff?|jpe?g|png|bmp)$', [input_label])
    dset.callbacks = callbacks
    dset.set_ignore_extensions(ignore_exts)

    # create a specific group for the labels and a spot to store the data
    label_group = dset.require_group(LABEL_PATH)
    data_group = dset.require_group(DATA_PATH)

    # scan the input folder and construct a dataset
    dset.make_dataset_from_folder(
        path=on_folder,
        group=data_group,
        recursive=True,
        ignore_invalid=True,
        label_group=label_group
    )

    return dset


def _on_make_file(out_dir: str, opt: dict, out_list: list, ignore_unmatched: bool = False):
    """ Called when we want to make an output dataset as well """
    if 'input' in opt['labels']:
        basic_path = opt['name_path'].replace('/' + DATA_PATH + '/', '')
        out_file = posixpath.join(out_dir, basic_path)
        if os.path.isfile(out_file + '.tif'):
            out_file += '.tif'
            has_file = True
        elif os.path.isfile(out_file + '.tiff'):
            out_file += '.tiff'
            has_file = True
        elif not ignore_unmatched:
            raise RuntimeError(
                'The file ' + out_file + ' does not exist for the in file of ' + basic_path + ' or is not a tiff')
        else:
            has_file = False

        # create the output dataset
        if has_file:
            # data = tifffile.imread(out_file) # NOTE: when profiling this was by far the slowest part (imread of tiff needs to be improved)
            out_name = posixpath.join(OUT_PATH, basic_path)
            if not out_name.startswith('/'):
                out_name = '/' + out_name

        # create label group if not already there
        if LABEL_PATH not in opt['dataset'].data:
            opt['dataset'].make_group(LABEL_PATH)

        # update the input file's attribute to include the output path
        in_obj = opt['dataset'].get(opt['name_path'])
        if in_obj is None:
            raise RuntimeError('Fatal error! Input object suddenly not found')

        if has_file:
            in_obj.attrs['output'] = out_name
        else:
            in_obj.attrs['output'] = ''

        depth = in_obj.attrs['depth']

        if has_file:
            # add the file to the output dataset and add the output label
            out_list.append((depth, out_name))

            # make the output dataset file
            _, name_path = opt['dataset'].make_dataset_from_file(
                out_file, out_name, add_labels=None, call_make_file_callback=False, depth=depth)

            # add the input key to the output file
            opt['dataset'].get(name_path).attrs['input'] = in_obj.name


def _on_make_group(opt: dict, out_data: dict):
    if str(opt['depth']) in out_data:
        out_data[str(opt['depth'])].append(opt['name_path'])
    else:
        out_data[str(opt['depth'])] = [opt['name_path']]


def create_input_output_dataset(on_folder: str, out_folder: str, file_path: str = 'dataset.h5', root_group: str = 'data',
        ignore_exts: list = ['json', 'txt', 'js', 'pdf'], callbacks: dict = {}, ignore_unmatched: bool = True, new_file: bool=True, keep_existing: bool=False) -> Dataset:
    if out_folder is None:
        raise ValueError('Output directory cannot be None')

    if not os.path.isdir(out_folder):
        raise RuntimeError('The output directory does not exist ' + out_folder)

    # custom handler for when a file is made to add an output file
    output_data = []
    output_groups = {}
    callbacks['make_file'] = lambda opt: _on_make_file(
        out_folder, opt, output_data, ignore_unmatched
    )
    callbacks['make_group'] = lambda opt: _on_make_group(
        opt, output_groups
    )

    dset = create_input_dataset(
        on_folder, file_path, root_group, 'input', ignore_exts, callbacks, new_file, keep_existing)

    # run each call on the dataset (these are processed one at a time)
    label_group = dset.get(LABEL_PATH)

    # create the groups
    depth_out_fgroup = label_group.require_group('depth_out_files')
    depth_out_ggroup = label_group.require_group('depth_out_groups')

    # deconstruct to add depths out as well
    if output_data:
        output_depths, output_labels = zip(*output_data)
        output_dict = {}
        for depth, label in output_data:
            if str(depth) in output_dict:
                output_dict[str(depth)].append(label)
            else:
                output_dict[str(depth)] = [label]

        # add both of the name lists to the groups
        dset.add_label_list('output', output_labels, label_group=label_group)

        for key in output_dict.keys():
            val = output_dict[key]
            dset.add_label_list(key, val, label_group=depth_out_fgroup)

        for key in output_groups.keys():
            val = output_groups[key]
            dset.add_label_list(key, val, label_group=depth_out_ggroup)

    return dset


if __name__ == '__main__':
    dset = create_input_output_dataset('C:\\Users\\smerk\\Documents\\Ground Truth Project\\Fabry - 09-0598\\09-0598 blk 1-1\\surs',
                                       'C:\\Users\\smerk\\Documents\\Ground Truth Project\\Fabry - 09-0598\\09-0598 blk 1-1\\out_class')
    print('done')
    # print('Testing dataset')
    # dset = Dataset('test.h5', create_new=True)
    # dset.set_custom_data_labeler('.*\.(tiff?|jpe?g|png|bmp)$', ['input'])
    # label_group = dset.make_group('labels')
    # dset.make_dataset_from_folder('C:\\Users\\smerk\\UW\\Najafian Lab - Lab Najafian\\Foot Process Workspace\\Ground Truth Project\\Fabry - 09-0598', dset, ignore_invalid=True, label_group=label_group)
    # data = dset.get_np('/09-0598 blk 3-1/surs/09--_4445')
    # print(list(dset.data.keys()))
    # print(label_group.keys())
    # print(dset.get(dset.get('labels/input')[:][0]))
    # cv2.imshow('test', data)
    # cv2.waitKey(0)
    # dset.close()
    # print('Done')
