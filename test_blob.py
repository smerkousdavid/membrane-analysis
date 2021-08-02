import unittest
from analysis import make_fpw_measurements
from tests.statistics import compare_statistics
from interactive.base import ResultLayer
from interactive.export import ImageExport
import numpy as np
import math
import os
import cv2

image = np.zeros((800, 800), dtype=np.uint8)
cv2.circle(image, (400, 400), 300, 255, 5)
cv2.circle(image, (400, 700), 5, 0, -1)  # cutout bottom for endpoint detection

# create background
back_layer = ResultLayer('background', 'Background')
back_layer.draw_image(image, (800, 800))

# circle left to top extent to right extent (so sum should be arc of pi and mean should be arc of pi/2)
points = np.array([
    [90, 400],  # left
    [400, 90],  # top
    [710, 400]   # right
], dtype=np.int32)

slits = np.zeros((800, 800), dtype=np.uint8)
for point in points:
    p = tuple(point)
    cv2.circle(slits, p, 2, 255, -1)

image = [image, slits]

blobs = np.ascontiguousarray(image)
blob_uint = (blobs > 50).astype(np.uint8)

# get the layers
membrane = blob_uint[-2]
slits = blob_uint[-1]

# make the measurements
fpw = make_fpw_measurements(membrane, slits, draw=True, export=True)

# do the export
exp = ImageExport()
exp.write_export('test.html', 'test image', [back_layer] + fpw.get_export())