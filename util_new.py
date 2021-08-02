import unittest
from structure.analysis.statistics import Statistics
from structure.grouped.analysis import GroupedAnalysis
from structure.interactive.dataset import Dataset, create_input_output_dataset
from openpyxl import Workbook
import tifffile
import shutil
import numpy as np
import math
import os
import time
import cv2
import json


def get_folders(path):
    items = os.listdir(path)
    return [os.path.join(path, item) for item in items if os.path.isdir(os.path.join(path, item))]

def get_files(path):
    items = os.listdir(path)
    return [os.path.join(path, item) for item in items if os.path.isfile(os.path.join(path, item))]


PROCESS = 'C:\\Users\\smerk\\Documents\\Ground Truth Project'
MAIN_FOLDERS = get_folders(PROCESS)
STRUCTURE = [get_folders(path) for path in MAIN_FOLDERS]
OUTPUT = 'out_class'
INPUT = 'surs'
INPUT_EXT = '.tif'
OUTPUT_EXT = '.tiff'
SAVE_OUTPUT = 'C:\\Users\\smerk\\Documents\\gtp_out'
PROC_IN = os.path.join(SAVE_OUTPUT, 'input')
PROC_OUT = os.path.join(SAVE_OUTPUT, 'output')
SCALING_DATA = {
    '0958-grid.tif': '9.914346895 nm/pixel',
    '13-grid.tif': '8.990291262 nm/pixel',
    '78-grid.tif': '9.625779626 nm/pixel'
}


def reorganize_structure():
    for main, subfolders in zip(MAIN_FOLDERS, STRUCTURE):
        print('Processing', main)
        for path in subfolders:
            ins = os.path.join(path, INPUT)
            outs = os.path.join(path, OUTPUT)

            # get the matched files
            inf = get_files(ins)
            outf = get_files(outs)

            # make sure they're matching
            if len(inf) != len(outf):
                print('miscount failure!', len(inf), len(outf), 'for', path)
                exit(1)
            
            rel_path = os.path.relpath(path, PROCESS)
            proc_in = os.path.join(PROC_IN, rel_path)
            proc_out = os.path.join(PROC_OUT, rel_path)

            # make relpath for in and out
            os.makedirs(proc_in)
            os.makedirs(proc_out)

            # copy each file over
            for i in inf:
                shutil.copy(i, proc_in)
            for o in outf:
                shutil.copy(o, proc_out)

def make_dataset():
    dset = create_input_output_dataset(PROC_IN, PROC_OUT, new_file=True, keep_existing=False, scaling_data=SCALING_DATA)
    sset = dset.save_to_scaled_dataset()
    # analysis = GroupedAnalysis(sset)
    # unprocessed = analysis.run()
    # # print(dset.get_all_ds_recursive(start='/analysis', name='slit_arc_distances'))
    # stats = analysis.get_stats_at_depth(0)
    # dset.close()
    # print(stats)
    # print('WARNING! The following files were not processed', unprocessed)

    # print(dset.get_all_ds_recursive(attribute='depth', attribute_equals=2))


def process_dataset():
    analysis = GroupedAnalysis('scaled_dataset.h5')
    analysis.run()

    stat_data = analysis.get_stats_at_depths([0, 1]) + analysis.get_stats_at_depths([2], stats_type='files')

    book = Workbook()
    for depth_data in stat_data:
        depth = depth_data['depth']
        stats = depth_data['stats']

        sheet = book.create_sheet(title='Depth %d' % int(depth))
        header = ['Name', 'Group', 'Units'] + Statistics.get_header_data()
        sheet.append(header)

        for stat in stats:
            sheet.append([stat['file'], stat['group'], stat['units']] + stat['slit_arc_distances'].get_row_data())
    book.save('test.xlsx')

# STEPS 1. reorganize structure 2. make the dataset 3. process dataset
# reorganize_structure()
# make_dataset()
process_dataset()