import unittest
from analysis.skeleton import fast_skeletonize
from analysis.hitmiss import scan_for_end
from analysis.treesearch import search_image_skeleton
from analysis.membrane import skeletons_to_membranes, measure_points_along_membrane
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

def checkIfDuplicates_2(listOfElems):
    ''' Check if given list contains any duplicates '''    
    setOfElems = set()
    for elem in listOfElems:
        if tuple(elem) in setOfElems:
            return True
        else:
            setOfElems.add(tuple(elem))         
    return False


class TestImageMeasures(unittest.TestCase):
    def _compare(self, file, show=False):
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

        # skeletonize the image
        skeleton = fast_skeletonize(blob_uint)

        # search for the branch endpoints
        end_points = scan_for_end(skeleton)

        # search the entire skeleton using the referenced endpoints
        skeleton_data = search_image_skeleton(skeleton, end_points, row_first=False)

        # convert skeleton data into membranes (this might take a while while searching the skeletons)
        membranes = skeletons_to_membranes(
            skeletons=skeleton_data,
            connect_close=1,  # connect membranes that are nearby
            padding=15,
            max_px_from_ends=30,
            max_angle_diff=((140.0 * math.pi) / 180.0),
            row_first=0
        )

        # copy threshold image and convert to color to draw on
        draw_on = cv2.cvtColor(blob_uint * 255, cv2.COLOR_GRAY2BGR)

        for mem in membranes:
            data = mem.get_points()
            # print('Arc Test', cv2.arcLength(data, False), dist[r_ind])
            # print('Direct Test', math.sqrt(((paired[ind][0][0]-paired[ind][1][0])**2) + ((paired[ind][0][1]-paired[ind][1][1])**2)), direct[ind])
            print(checkIfDuplicates_2(data))
            points = np.append(data, data[::-1], axis=0)
            cv2.drawContours(draw_on, [points], -1, (0, 255, 0), 2)

        # show results
        if show:
            cv2.imshow('File %s (draw on)' % file, cv2.resize(draw_on, (500, 500)))
            cv2.imshow('File %s (skeleton)' % file, cv2.resize(skeleton.astype(np.uint8) * 255, (500, 500)))
            cv2.waitKey(WAITKEY)

    def test_connection(self):
        self._compare(
            'connection.png',
            show=True
        )
    
    def ftest_near_pair(self):
        # line 1 (straight across)
        image = np.zeros((300, 300), dtype=np.uint8)
        cv2.line(image, (10, 150), (135, 150), 255, 5)
        cv2.line(image, (160, 150), (290, 150), 255, 5)

        self._compare(
            image,
            show=True
        )

        # line 2 (at an ngle)
        image = np.zeros((300, 300), dtype=np.uint8)
        cv2.line(image, (10, 150), (135, 150), 255, 5)
        cv2.line(image, (160, 170), (290, 170), 255, 5)

        self._compare(
            image,
            show=True
        )

        # line 3 (right above)
        image = np.zeros((300, 300), dtype=np.uint8)
        cv2.line(image, (10, 150), (135, 150), 255, 5)
        cv2.line(image, (135, 180), (290, 180), 255, 5)

        self._compare(
            image,
            show=True
        )

        # line 4 (at a steeper close angle)
        image = np.zeros((300, 300), dtype=np.uint8)
        cv2.line(image, (10, 150), (135, 150), 255, 5)
        cv2.line(image, (110, 160), (290, 160), 255, 5)

        self._compare(
            image,
            show=True
        )

        # line 5 (at a steeper close above angle)
        image = np.zeros((300, 300), dtype=np.uint8)
        cv2.line(image, (10, 150), (135, 150), 255, 5)
        cv2.line(image, (110, 140), (290, 140), 255, 5)

        self._compare(
            image,
            show=True
        )