from typing import List, Tuple
import json
import multiprocessing
import os
import posixpath
import queue
import math
import re
import threading

from structure import ureg, Q_
from structure.image import resize_image_nparray, image_dims
import cv2
import shutil
import h5py as hf
import numpy as np
import tifffile
import pint

# string datatype for h5 files
HF_STR = hf.string_dtype(encoding='utf-8')
LABEL_PATH = 'labels'
DATA_PATH = 'data'
OUT_PATH = 'out'
TARGET_DIMENSION = '8nm/pixel'  # very important unit as this will attempt to rescale images so that each pixel is 8nm
PIXEL = 'pixel'

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


def fix_scale_path(path: (str, bytes)) -> str:
    if isinstance(path, bytes):
        path = path.decode('utf-8')
    path = str(path).lower().strip()
    if len(path) > 1:  # check for windows variants
        if path[1:].startswith(':\\'):  # drive name removal
            path = path[2:]  # remove windows drive name portion

    return path.replace('\\', '/')


class Dataset(object):
    def __init__(self, file: str, create_new: bool = False, color_to_grayscale: bool=True, interpolation='lancz'):
        if not os.path.isfile(file):
            create_new = True  # we have to write a new h5 file
        self.data = hf.File(file, 'w' if create_new else 'a')

        # default scaling data if not defined
        if 'scaled' not in self.data.attrs and 'scaling' not in self.data.attrs:
            self.data.attrs['scaled'] = False
            self.data.attrs['scaling'] = 'NONE'
        self.data_loader = {}
        self.data_labels = {}
        self.data_functions = {}
        self.callbacks = {}
        self.scaling_data = None
        self.default_scaling_type = None
        self.default_grid_size = None
        self.keep_existing = False
        self.color_to_grayscale = color_to_grayscale
        self.ignore_extensions = set()
        self.interpolation = interpolation
        self.threads = multiprocessing.cpu_count()
        
        # new files are always dirty in terms of analysis
        if create_new:
            self.set_dirty(True)
        self.create_new = create_new

        # populate the default image loader
        self.set_custom_data_loader(self.__read_cv2, ['png', 'jpg', 'jpeg', 'bmp'])
        self.set_custom_data_loader(self.__read_tiff, ['tif', 'tiff'])

    def __read_cv2(self, path):
        return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if self.color_to_grayscale else cv2.IMREAD_UNCHANGED)
    
    def __read_tiff(self, path):
        image = tifffile.imread(str(path))
        if image is not None and self.color_to_grayscale and len(image.shape) == 3 and image.shape[-1] == 3:  # it's a color image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
 
    def is_dirty(self) -> bool:
        if 'dirty' not in self.data.attrs:
            self.data.attrs['dirty'] = True  # update the file (could be used for later data)
        return self.data.attrs['dirty']

    def is_scaled(self) -> bool:
        return self.data.attrs.get('scaled', False)
    
    def get_dataset_scaling(self) -> pint.Quantity:
        if self.is_scaled():
            d = self.data.attrs.get('scaling', None)
            if d is None:
                return None
            return Q_(d)
        return None

    def set_dirty(self, dirty: bool=True):
        self.data.attrs['dirty'] = dirty

    def set_threads(self, num: int):
        self.threads = num

    def set_keep_existing(self, existing: bool):
        self.keep_existing = existing

    def save_to_scaled_dataset(self, file_name: str='scaled_dataset.h5', target_dimension: (str, pint.Quantity)=TARGET_DIMENSION, labels: List[str]=['input', 'output']):
        # flush and copy file
        self.data.flush()
        fname = self.data.filename
        self.data.close()

        # copy dataset
        shutil.copy(fname, file_name)

        # reopen original
        self.data = hf.File(fname, 'a')

        # now create new scaled reference dataset and copy some of the basic properties/loaders over
        scaled = Dataset(file_name, create_new=False, color_to_grayscale=self.color_to_grayscale, interpolation=self.interpolation)
        scaled.callbacks = self.callbacks
        scaled.data_functions = self.data_functions
        scaled.data_labels = self.data_labels
        scaled.data_loader = self.data_loader
        scaled.default_grid_size = self.default_grid_size
        scaled.default_scaling_type = self.default_scaling_type
        scaled.ignore_extensions = self.ignore_extensions
        scaled.scaling_data = self.scaling_data

        # save the targeted dimension
        self.target_dimension = target_dimension if isinstance(target_dimension, pint.Quantity) else Q_(target_dimension)
        scaled.target_dimension = self.target_dimension
        scaled.data.attrs['scaled'] = True
        scaled.data.attrs['scaling'] = str(self.target_dimension)

        # get all the objects labeled within the paths and combine the results
        for label in labels:
            paths = self.get_str(posixpath.join(LABEL_PATH, label))
            
            # apply for all datasets
            for path in paths:
                # capture the object
                obj = scaled.get(path)
                if obj is None:
                    raise RuntimeError('The labeled object ' + path + ' was not found or is None!')

                # get attributes
                failed = False
                if 'scaled' not in obj.attrs:
                    failed = True
                elif obj.attrs['scaled']:
                    failed = True
                elif 'scaling' not in obj.attrs:
                    failed = True
                elif 'pixel' not in obj.attrs['scaling']:
                    failed = True
                
                # if we failed raise an error saying the specifed object isn't scaled
                if failed:
                    raise RuntimeError('There was not a valid scaling found (UNIT / pixel) for the image/data ' + path + ' or the image has already been scaled')

                # capture the scaling number (in the same units as the target)
                scaling = Q_(obj.attrs['scaling']).to(self.target_dimension.units)

                # now let's get the magnitude of the scaling
                scaling = float((scaling / self.target_dimension).magnitude)

                # DEBUG
                # print('Rescaling', scaling, 'from', obj.attrs['scaling'], 'to', target_dimension)

                # let's convert the data
                img = np.array(obj[:])
                
                # make new dims
                ndims = image_dims(img)
                ndims = (round(float(ndims[0]) * scaling), round(float(ndims[1] * scaling)))

                # make the new image to match the target resolution dimensions
                resized = resize_image_nparray(img, ndims, interp=self.interpolation, fix_depth=True)

                # update the h5py dataset
                new_props = dict(obj.attrs)

                # update scaling information
                old_scaling = new_props['scaling']
                new_props.update({
                    'scaled': True,
                    'original_scaling': old_scaling,
                    'scaling': str(self.target_dimension)
                })

                # remove current dataset
                del scaled.data[path]

                # create the new resized version
                ndset = scaled.data.create_dataset(path, shape=resized.shape, dtype=resized.dtype, data=resized)
                
                # copy attributes from old dataset
                self.__copy_props(ndset, new_props)
        
        return scaled                 

    def set_scaling_data(self, data: dict, default_scaling_type:(str, pint.Quantity)='nm/pixel', default_grid_size:(str, pint.Quantity)=460):
        if data is None:
            self.scaling_data = None
            return

        self.scaling_data = {}
        self.default_scaling_type = default_scaling_type
        self.default_grid_size = default_grid_size

        # let's fix it up by fixing keys to be matchable
        for key in data.keys():
            val = data[key]

            # make sure it's a proper scaling factor
            if val is not None:
                if isinstance(val, (int, float)):
                    val = Q_(str(val), default_scaling_type)  # dimensionless data doesn't have a scaling factor (use a default)
                elif isinstance(val, (str, bytes)):
                    oval = val
                    if isinstance(val, bytes):
                        val = val.decode('utf-8')
                    val = Q_(val)

                    # make sure it has a dimension
                    if str(val.units) == 'dimensionless':
                        print('WARNING a dimensionless string was passed into a scaling factor {}. Using default of {}'.format(str(oval), default_scaling_type))
                        val = Q_(oval, default_scaling_type)
                # elif isinstance(val, np.ndarray):
                elif not isinstance(val, pint.Quantity):
                    raise ValueError('Provided value for scaling/calibration is not a valid type {}'.format(str(val)))

                # let's determine if the scaling data is either a grid size, actual conversion, or something else
                is_conversion_factor = str(val.units).replace(' ', '').endswith('/pixel')
                self.scaling_data[fix_scale_path(key)] = {
                    'scaling': is_conversion_factor,
                    'value': val
                }

    def __copy_props(self, item: (hf.Group, hf.Dataset), props: dict):
        """ Simple function to update the attributes of a object

        Args:
            item ([Group or Dataset]): item to copy attribute properties to
            props (dict): all properties to copy over
        """
        for key in props.keys():
            item.attrs[key] = props[key]

    def __recurse_apply_props(self, group: hf.Group, props: dict, match_label: (str, bytes), include_groups: bool=True):
        """ Applies the properties to all values starting at the specified group (sub)

        Args:
            group (hf.Group): group to start the property application
            props (dict): the attributes to copy over to the object
            include_groups (bool): also apply properties to all sub-groups
        """
        if group is None or isinstance(group, hf.Dataset) or props is None:
            return
        
        # labels are utf-8 byte encoded
        if isinstance(match_label, str):
            match_label = match_label.encode('utf-8')

        for key in group.keys():
            item = group[key]

            # check types
            if isinstance(item, hf.Dataset) and match_label in item.attrs['labels']:
                self.__copy_props(item, props)
            elif isinstance(item, hf.Group):
                if include_groups:
                    self.__copy_props(item, props)

                # now recurse over the new group
                self.__recurse_apply_props(item, props, match_label, include_groups)
            elif not isinstance(item, (hf.Dataset, hf.Group)):
                raise RuntimeError('Invalid scanned object ' + key + ' type ' + str(item) + ' at ' + group.name)

    def recurse_scaling_data(self):
        """ Should only be called if set_scaling_data was called and the dataset has been created on a folder. This data is NOT PERSISTENT!
        
        NON-PERSISTENCY means that this function must be called if scaling values of data are wanted to be applied to the attributes in the h5 file
        """
        if self.scaling_data is None:
            return
        elif len(self.scaling_data) == 0:
            return
        elif self.default_grid_size is None:
            return
        
        # get the labeled files of the calibration grids
        calibs = self.get_str(posixpath.join(LABEL_PATH, 'calibration'))
        
        # make sure we have data
        if calibs is not None and len(calibs) > 0:
            for calib in calibs:
                calib = self.data[calib]
                parent = calib.parent

                # with the parent group now identified let's capture its attributes
                units = calib.attrs['units']
                scaling = calib.attrs['scaling']
                scaled = calib.attrs['scaled']

                # make the updates props dict
                props = {
                    'units': units,
                    'scaling': scaling,
                    'scaled': bool(scaled)
                }

                # let's recursively apply those properties to all sub-folders
                cname = str(parent.name)
                if cname.endswith('/' + DATA_PATH):
                    cname += '/'
                out_name = cname.replace('/' + DATA_PATH + '/', '/' + OUT_PATH + '/')
                self.__recurse_apply_props(parent, props, 'input', True)

                if out_name in self.data:
                    out_path = self.data[out_name]  # we want to do the same thing for the output labels
                    self.__recurse_apply_props(out_path, props, 'output', True)
                else:
                    print('WARNING no output name of', out_name, 'not applying dataset properties')

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
            self.set_dirty()

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
        self.set_dirty()

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

    def set_custom_data_analysis_function(self, regex: str, method):
        if method is None or regex is None:
            raise ValueError('method cannot be None')
        
        # set labeling based off of analysis
        self.data_functions[regex] = method

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
        return None if len(labels) == 0 else np.array(labels, dtype='S')

    def _get_custom_file_labels(self, path: str, dset: hf.Dataset, data: object):
        # try to match all of the files
        f_name = get_file_name(path)
        f_ext = get_file_ext(path)

        # a normalized full file name
        name = os.path.join(os.path.dirname(
            path).lower(), f_name + '.' + f_ext)
        
        # do a analysis if any match the custom data anlysis labels
        if isinstance(data, np.ndarray):
            # attempt to find a corresponding label
            labels = []
            any_matched = False
            for key in self.data_functions.keys():
                if re.match(key, name) is not None:
                    ret = self.data_functions[key](self, name, dset, data)
                    if ret is not None:
                        ret, failed = ret

                        # if failure then skip this file as something went wrong in determining labels
                        if not failed:
                            any_matched = True  # follow current regex rule

                            # track label sets
                            if ret:
                                if isinstance(ret, (list, set)):
                                    if len(ret) >= 1 and isinstance(ret[0], bytes):
                                        ret = list(map(lambda x: x.decode('utf-8'), ret))
                                    labels.extend(ret)
                                else:
                                    if isinstance(ret, bytes):
                                        ret = ret.decode('utf-8')
                                    labels.append(ret)

            # one of them matched so don't return the labels just off of the file path
            if any_matched:
                return None if len(labels) == 0 else np.array(labels, dtype='S')

        # otherwise return the labels that are ONLY associated with the path
        return self._get_file_labels_from_path(path)

    def make_dataset_from_file(self, path: str, group: (hf.File, hf.Group, str) = None, cast_dtype: str = None, metadata: dict = None, custom_name: str = None, add_labels: list = None, label_group: (hf.Group) = None, call_make_file_callback: bool = True, depth: int = None, _rec: bool = False, _force_labels: List[str]=None, **kwargs) -> list:
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
        labels = None
        name_path = None

        if ext.replace('.', '').lower() in self.ignore_extensions:
            raise NotImplementedError(
                'The following file in in the ignored file list ' + path)

        if ext in self.data_loader:
            if self.keep_existing and ((group in self.data) if isinstance(group, str) else (name in group)):
                return None, None

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
            labels = np.array(_force_labels, dtype='S') if _force_labels else self._get_custom_file_labels(path, dset, data) # self._get_file_labels_from_path(path)
            dset.attrs['labels'] = hf.Empty('S1') if labels is None else labels

            if not _rec:  # not a recursive call like in make dataset from folder
                self.set_dirty()

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

    def add_label_list(self, label: (bytes, str), data: (list, set, tuple, np.ndarray), label_group: (hf.File, hf.Group), _rec: bool=False):
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
        
        if not _rec:
            self.set_dirty()

    def get_all_ds_recursive(self, start: (hf.File, hf.Group, hf.Dataset, bytes, str) = None, attribute: (str, bytes) = None, attribute_equals: (str, int, bytes) = None, name: (str, bytes) = None):
        if start is None:
            start = self.data
        elif isinstance(start, (bytes, str)):
            start = self.data[start]

        # let's start scanning recursively
        results = []
        if isinstance(start, hf.Group):
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
        elif isinstance(start, hf.Dataset):
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
        else:
            raise RuntimeError('Invalid object type at path ' + start.name)
        return results

    def __labeled_dataset_file_consumer(self, queue: queue.Queue, output: queue.Queue):
        while True:
            item = queue.get()

            # quit the worker
            if item is None:
                break

            try:
                labels, name_path = self.make_dataset_from_file(
                    item['full'], item['group'], depth=item['depth'], _rec=True)

                if labels is not None and name_path is not None:
                    # array of labels not an empty dataset
                    if not isinstance(labels, hf.Empty) and labels is not None:
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
            # we'll mark the file as dirty
            self.set_dirty()

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
                if isinstance(item, dict):
                    if (not ignore_invalid) or item['fatal']:
                        raise item['error']
                    elif ignore_invalid:
                        print('WARNING: Ignorable error thrown in thread ERROR: "{}"'.format(item['error']))
                        continue
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
            for val in tracked_labels.keys():
                if val:
                    all_labels.add(val)
            
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


def parse_scaling_loosely(dataset: Dataset, scaling_data: dict, path: str) -> dict:
    # attempts to parse the provided scaling data progressively until a match is made
    if path is None or scaling_data is None:
        return None  # no matching

    # first try an exact match
    fpath = fix_scale_path(path)  # for win convert paths to posix
    if fpath in scaling_data:
        return scaling_data[fpath]

    # now let's try a subset of paths (such as the directory name)
    keys = scaling_data.keys()
    dired = posixpath.dirname(fpath)
    name = posixpath.basename(fpath)

    # test the secondary sets
    for test in keys:
        if dired == posixpath.dirname(test):
            return scaling_data[test]
        elif name == posixpath.basename(test):
            return scaling_data[test]

    # tertiary sets where we look at only relative paths
    for test in keys:
        if test in fpath:  # it's a subset of it
            return scaling_data[test]

    # finally see if there is some hint of scaling in the file name (only supporting nm at the moment)
    if 'grid' in name and 'nm' in name:
        # @TODO FIX REGEX
        match = re.match(r'.*(([0-9]){1:10}nm).*', name, re.M)
        if match:
            num = float(match.group(1)) # @TODO FIX GROUP
            return {
                'scaling': False,  # not a scaling just a grid size
                'value': Q_(num, 'nm')  # set the scaling value
            }
    
    return None  # couldn't find any scaling data


def grid_analysis(dataset: Dataset, path: str, dset: hf.Dataset, data: np.ndarray) -> Tuple[List[str], bool]:
    # attempt to use the already defined scaling data (ignore multi-layer objects as those aren't inputs)
    # @TODO make something more robut for future that could accept 3D slices
    if len(data.shape) > 2 and data.shape[-1] > 3:  # not a color image but rather a series of images
        return ['output'], False 

    grid_size = dataset.default_grid_size
    scaling_factor = None
    if dataset.scaling_data:
        scaling = parse_scaling_loosely(dataset, dataset.scaling_data, path)
        
        # determine if we have a scaling factor or not
        if scaling is not None:
            if scaling['scaling']:
                scaling_factor = scaling['value'] # we have a valid scaling factor
            else:
                grid_size = scaling['value']  # we have a valid grid size to try to automatically process

    # otherwise try to analyze the image and determine if it's a grid or not (accounting for a default grid size)
    if scaling_factor is None and grid_size is not None:
        # DO FANCY GRID ANALYSIS STUFF HERE!
        # raise NotImplementedError('Currently cannot do anything for this')
        pass
    
    # let's finally do something with a valid scaling factor (such as add the valid labels)
    pg = dset.parent  # get the parent group
    if scaling_factor is None:
        dset.attrs['units'] = 'pixel'
        dset.attrs['scaled'] = False

        # apply similar attributes to parent group (if a calibration set has not been set yet)
        if 'units' not in pg.attrs:
            pg.attrs['units'] = 'pixel'
            pg.attrs['scaled'] = False
        return ['input'], False
    else:
        units = ((scaling_factor if isinstance(scaling_factor, pint.Unit) else scaling_factor.units) * Q_('pixel').units)  # remove conversion factor unit of pixel
        units = str(units if isinstance(units, pint.Unit) else units.units)
        scaling = str(scaling_factor)
        dset.attrs['units'] = units
        dset.attrs['scaling'] = scaling
        dset.attrs['scaled'] = False

        # apply similar attributes to parent group
        pg.attrs['units'] = units
        pg.attrs['scaling'] = scaling
        pg.attrs['scaled'] = False
        return ['calibration'], False

def create_input_dataset(on_folder: str, file_path: str = 'dataset.h5', root_group: str = 'data', input_label='input',
            ignore_exts: list = ['json', 'txt', 'js', 'pdf'], callbacks: dict = {}, new_file: bool=True, keep_existing: bool=False, scaling_data: dict=None, ignore_names=['prog', 'out', 'out_full']) -> Dataset:
    if on_folder is None:
        raise ValueError('The folder cannot be None')

    if not os.path.isdir(on_folder):
        raise RuntimeError('The folder ' + on_folder + ' does not exist')

    # create the new default dataset
    dset = Dataset(file_path, create_new=new_file)
    dset.set_keep_existing(keep_existing)
    dset.set_scaling_data(scaling_data)
    # dset.set_custom_data_labeler(r'.*\.(tiff?|jpe?g|png|bmp)$', [input_label])
    
    # create the grid analysis labeling
    dset.set_custom_data_analysis_function(r'.*\.(tiff?|jpe?g|png|bmp)$', grid_analysis)
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
    dset.recurse_scaling_data()  # apply the scaled data to the sub-datasets
    dset.set_dirty()  # just to be sure

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
                out_file, out_name, add_labels=None, call_make_file_callback=False, depth=depth, _force_labels=['output'])

            # add the input key to the output file
            opt['dataset'].get(name_path).attrs['input'] = in_obj.name


def _on_make_group(opt: dict, out_data: dict):
    if str(opt['depth']) in out_data:
        out_data[str(opt['depth'])].append(opt['name_path'])
    else:
        out_data[str(opt['depth'])] = [opt['name_path']]


def create_input_output_dataset(on_folder: str, out_folder: str, file_path: str = 'dataset.h5', root_group: str = 'data',
        ignore_exts: list = ['json', 'txt', 'js', 'pdf'], callbacks: dict = {}, ignore_unmatched: bool = True, new_file: bool=True, keep_existing: bool=False, scaling_data: dict=None) -> Dataset:
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
        on_folder, file_path, root_group, 'input', ignore_exts, callbacks, new_file, keep_existing, scaling_data=scaling_data)

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
                                       'C:\\Users\\smerk\\Documents\\Ground Truth Project\\Fabry - 09-0598\\09-0598 blk 1-1\\out_class',
                                       scaling_data={
                                           'ground truth project\\fabry - 09-0598\\09-0598 blk 1-1\\surs\\09--_4339.tif': '10 nm/pixel',
                                       })
    
    # creates a new file for the scaled dataset and copies the data over to a new Dataset object
    sset = dset.save_to_scaled_dataset()
    

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
