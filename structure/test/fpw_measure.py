import unittest
from analysis import make_fpw_measurements
from tests.statistics import compare_statistics
import numpy as np
import math
import os
import cv2


PY_PATH = os.path.dirname(os.path.realpath(__file__))
IMAGE_FOLDER = os.path.join(PY_PATH, 'test_imgs')
TEST_IMGS = [os.path.join(IMAGE_FOLDER, f).lower() for f in os.listdir(IMAGE_FOLDER) if 'png' in os.path.splitext(f)[1].lower()]
MAX_DIFF = 0.1  # max difference in distances/pixel numbers
THRESHOLD = 50  # min value in image to turn into a binary 1
WAITKEY = -1


class TestFPWMeasures(unittest.TestCase):
    def _compare(self, file, res, show=False):
        # make sure file exists (unless if it's a numpy array)
        if isinstance(file, (list, tuple, np.ndarray)):
            image = file
        else:
            fpath = os.path.join(IMAGE_FOLDER, file).lower()
            if fpath not in TEST_IMGS:
                raise ValueError('The image file %s does not exist in the test suite!' % file)

            image = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        blobs = np.ascontiguousarray(image)
        blob_uint = (blobs > 50).astype(np.uint8)

        # get the layers
        membrane = blob_uint[-2]
        slits = blob_uint[-1]

        # make the measurements
        fpw = make_fpw_measurements(membrane, slits, draw=show)
        
        # compare the stats
        compare_statistics(fpw.get_arc_stats().json(), res)

        # show results
        if show and fpw.has_image():
            cv2.imshow('File %s (draw on)' % file, cv2.resize(fpw.get_image(), (500, 500)))
            cv2.waitKey(WAITKEY)

    def ftest_disconnected_pair(self):
        image = np.zeros((300, 300), dtype=np.uint8)
        cv2.line(image, (10, 150), (135, 150), 255, 5)
        cv2.line(image, (160, 150), (290, 150), 255, 5)
        
        points = np.array([
            [20, 150],  # left
            [280, 150],  # right
        ], dtype=np.int32)

        self._compare(
            image,
            points,
            [
                {
                    'mean': 259.8284271247462,
                },
                {}
            ]
        )

    def test_drawn_circle(self):
        image = np.zeros((800, 800), dtype=np.uint8)
        cv2.circle(image, (400, 400), 300, 255, 5)
        cv2.circle(image, (400, 700), 5, 0, -1)  # cutout bottom for endpoint detection

        # circle left to top extent to right extent (so sum should be arc of pi and mean should be arc of pi/2)
        points = np.array([
            [90, 400],  # left
            [400, 90],  # top
            [710, 400]   # right
        ], dtype=np.int32)

        # compute pi/2 arc
        arc = 495.90158

        slits = np.zeros((800, 800), dtype=np.uint8)
        for point in points:
            p = tuple(point)
            cv2.circle(slits, p, 2, 255, -1)

        self._compare(
            [image, slits],
            {
                'sum': 2.0 * arc,
                'mean': arc,
            }
        )