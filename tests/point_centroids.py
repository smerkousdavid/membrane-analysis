import unittest
from analysis import get_slit_locations
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


class TestImageCentroid(unittest.TestCase):
    def _compare(self, file, points, show=False):
        # make sure file exists (unless if it's a numpy array)
        if isinstance(file, np.ndarray):
            image = file
        else:
            fpath = os.path.join(IMAGE_FOLDER, file).lower()
            if fpath not in TEST_IMGS:
                raise ValueError('The image file %s does not exist in the test suite!' % file)

            image = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        blobs = np.ascontiguousarray(image)
        blob_uint = (blobs > 50).astype(np.uint8)

        # copy threshold image and convert to color to draw on
        draw_on = cv2.cvtColor(blob_uint * 255, cv2.COLOR_GRAY2BGR)

        # make sure the results are the same
        found = set([tuple(t) for t in get_slit_locations(blob_uint)])
        expected = set([tuple(t) for t in points])

        # show results
        if show:
            for t in found:
                cv2.circle(draw_on, tuple(t), 3, (255, 0, 0), -1)
            for t in expected:
                cv2.circle(draw_on, tuple(t), 3, (0, 255, 0), -1)
            cv2.imshow('File %s (draw on)' % file, cv2.resize(draw_on, (500, 500)))
            cv2.waitKey(WAITKEY)

        # make sure they're good
        assert found == expected, 'Mismatched points found'

    def test_circles(self):
        image = np.zeros((500, 500), dtype=np.uint8)
        points = np.zeros((5, 2), dtype=np.int32)

        for ind, p in enumerate([(30, 30), (10, 20), (400, 400), (250, 250), (100, 20)]):
            cv2.circle(image, p, 3, 255, -1)
            points[ind, :] = p
        
        self._compare(
            image,
            points
        )
