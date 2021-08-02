import unittest
from analysis.statistics import Statistics
from analysis import make_fpw_measurements
from tests.statistics import compare_statistics
from interactive.base import ResultLayer
from interactive.export import ImageExport
from interactive.dataset import create_input_dataset
from grouped.analysis import GroupedAnalysis
from openpyxl import Workbook
import tifffile
import numpy as np
import math
import os
import time
import cv2


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
exp = ImageExport()


def process_single(input, output):
    print('Image', output)
    in_image = cv2.imread(input, cv2.IMREAD_GRAYSCALE)
    layers = tifffile.imread(output)

    layers = np.ascontiguousarray(layers)
    layers_uint = (layers > 50).astype(np.uint8)
    membrane = layers_uint[-2]
    slits = layers_uint[-1]

    # make the measurements
    fpw = make_fpw_measurements(membrane, slits, draw=False, export=True)

    back_layer = ResultLayer('background', 'Background')
    back_layer.draw_image(in_image, in_image.shape[::-1])
    
    exports = [back_layer]
    if fpw.is_valid():
        exports += fpw.get_export()
    
    basic = os.path.splitext(os.path.basename(input))[0]
    exp.write_export(os.path.join(SAVE_OUTPUT, basic + '.html'), os.path.basename(input), exports)
    return fpw


def process_biopsy(path):
    in_loc = os.path.join(path, INPUT)
    out_loc = os.path.join(path, OUTPUT)
    ins = get_files(in_loc)
    outs = get_files(out_loc)

    if len(ins) != len(outs):
        print('miscount failure!', len(ins), len(outs))
        exit(1)

    based = [os.path.splitext(os.path.basename(n))[0] for n in ins]
    singles = []
    for base in based:
        in_file = os.path.join(in_loc, base + INPUT_EXT)
        out_file = os.path.join(out_loc, base + OUTPUT_EXT)

        if not os.path.isfile(in_file) or not os.path.isfile(out_file):
            print('input or outputs are not files!', in_file, out_file)
            exit(1)
        
        singles.append(process_single(in_file, out_file))
    return based, singles


# book = Workbook()
# for name, main in zip(MAIN_FOLDERS, STRUCTURE):
#     sheet = book.create_sheet(title=os.path.basename(name))
#     for biopsy in main:
#         print('Processing', biopsy)
#         sheet.append([])
#         sheet.append(['BIOPSY ' + os.path.basename(biopsy)])
#         sheet.append(['Name'] + Statistics.get_header_data())
#         all_res = process_biopsy(biopsy)
#         for name_file, res in zip(*all_res):
#             if res.is_valid():
#                 sheet.append([name_file] + res.get_row_data())
# book.save('test.xlsx')

dset = create_input_dataset('')