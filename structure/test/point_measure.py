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


class TestImageMeasures(unittest.TestCase):
    def _compare(self, file, measure_points, res, show=False):
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
            padding=10,
            max_px_from_ends=50,
            max_angle_diff=((15.0 * math.pi) / 180.0),
            row_first=0
        )

        # make the point measurements using the membranes
        measures = measure_points_along_membrane(
            image=blob_uint,
            membranes=membranes,
            points=measure_points,
            max_px_from_membrane=500,
            density=0.1,
            min_measure=3,
            measure_padding=10,
            max_measure_diff=1.2
        )

        # copy threshold image and convert to color to draw on
        draw_on = cv2.cvtColor(blob_uint * 255, cv2.COLOR_GRAY2BGR)

        # make sure we're tesing the same amount of membranes
        assert isinstance(res, (tuple, list)), 'Expected result must be a tuple or list'
        # assert len(res) == len(measures), 'Difference in count between result and data EXPECTED: %d and GOT: %d' % (len(res), len(measures))

        # "pair all membrane stats" by comparing their excpetions
        mem_ids = []
        for mid, mes in enumerate(measures):
            for ind, exp in enumerate(res):
                try:
                    stats = mes.get_stats()
                    compare_statistics(stats.json(), exp)
                    mem_ids.append((mid, ind))

                    # matched so let's draw the results on our image
                    if show:
                        paired = mes.get_point_membrane_pairs()
                        rrange = mes.get_membrane_ranges()
                        dist = mes.get_arc_distances()
                        direct = mes.get_direct_distances()
                        mem_points = membranes[mid].get_points()
                        print('Entire Distance Test', cv2.arcLength(mem_points, False), membranes[mid].get_distance())

                        # valid points
                        if len(paired) > 0:
                            for r_ind, pair in enumerate(paired):
                                l1, l2 = pair
                                cv2.line(draw_on, tuple(l1), tuple(l2), (255, 0, 255), 1)
                            
                            for r_ind, rr in enumerate(rrange):
                                start, end = int(rr[0]), int(rr[1])
                                data = mem_points[start:end]
                                print('Arc Test', cv2.arcLength(data, False), dist[r_ind])
                                print('Direct Test', math.sqrt(((paired[ind][0][0]-paired[ind][1][0])**2) + ((paired[ind][0][1]-paired[ind][1][1])**2)), direct[ind])
                                points = np.append(data, data[::-1], axis=0)
                                cv2.drawContours(draw_on, [points], -1, (0, 255, 0), 2)
                        else:
                            print('No points!')
                except (IndexError, ValueError) as err:
                    print(err)
                    pass  # continue scanning
        
        # show results
        if show:
            cv2.imshow('File %s (draw on)' % file, cv2.resize(draw_on, (500, 500)))
            cv2.imshow('File %s (skeleton)' % file, cv2.resize(skeleton.astype(np.uint8) * 255, (500, 500)))
            cv2.waitKey(WAITKEY)

        # make sure they're the same count
        if len(mem_ids) != len(res):
            ids = set(list(range(len(measures))))
            found = set([i[0] for i in mem_ids])
            diff = ids - found

            # print diffs
            print()
            for i in diff:
                print('Membrane %d: %s' % (i, str(measures[i].get_stats())))
                print('    DISTANCES: %s' % str(measures[i].get_direct_distances()))

            raise ValueError('Difference in output of matched membrane ids! EXPECTED %d, GOT %d' % (len(res), len(mem_ids)))

    def test_square(self):
        points = np.array([
            [150, 250],
            [180, 250]
        ], dtype=np.int32)

        self._compare(
            'square.png',
            points,
            [
                {
                    'mean': 29,
                }
            ]
        )

    def test_square_ends(self):
        points = np.array([
            [10, 250],
            [400, 250]
        ], dtype=np.int32)

        self._compare(
            'square.png',
            points,
            [
                {
                    'sum': 245.41421,
                    'mean': 245.41421
                }
            ]
        )

    def test_disconnected_pair(self):
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

        self._compare(
            image,
            points,
            [
                {
                    'sum': 2.0 * arc,
                    'mean': arc,
                }
            ]
        )