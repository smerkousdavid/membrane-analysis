""" Contains all of the general analysis functions """
from typing import List
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

# local
from interactive.base import ResultLayer
from analysis.skeleton import fast_skeletonize
from analysis.hitmiss import scan_for_end
from analysis.treesearch import search_image_skeleton
from analysis.membrane import MembraneMeasureResults, skeletons_to_membranes, measure_points_along_membrane
from analysis.statistics import Statistics, compute_statistics


def average_points(pt1, pt2):
    return round((pt1[0] + pt2[0]) / 2.0), round((pt1[1] + pt2[1]) / 2.0)


class FootProcessWidths(object):
    def __init__(self):
        self.overall_arc = None
        self.overall_direct = None
        self.membrane_stats = []
        self.result_image = None
        self.export_result = None
        self.valid = True
        self.data = None
    
    def set_invalid(self):
        self.valid = False
    
    def is_valid(self) -> bool:
        return self.valid

    def set_data(self, data):
        self.data = data

    def get_data(self):
        return self.data

    def set_membrane_stats(self, all_stats: List[MembraneMeasureResults]):
        self.membrane_stats = all_stats

    def add_membrane_stats(self, membrane: MembraneMeasureResults):
        self.membrane_stats.append(membrane)

    def has_image(self) -> bool:
        return self.result_image is not None

    def has_export(self) -> bool:
        return self.export_result is not None

    def get_image(self) -> np.ndarray:
        return self.result_image

    def get_export(self) -> List[ResultLayer]:
        return self.export_result

    def set_image(self, image: np.ndarray):
        self.result_image = image

    def set_export(self, res: (ResultLayer, List[ResultLayer])):
        if res is None:
            self.export_result = None
        else:
            if isinstance(res, (tuple, list)):
                self.export_result = res
            else:
                self.export_result = [res]

    def compute_overall(self):
        measures_arc = []
        measures_direct = []
        for mes in self.membrane_stats:
            if not mes.is_empty():
                measures_arc.append(mes.get_arc_distances())
                measures_direct.append(mes.get_direct_distances())
        
        # compile results
        measures_arc = np.concatenate(measures_arc, axis=0).astype(np.double) if len(measures_arc) > 0 else np.array([], dtype=np.double)
        measures_direct = np.concatenate(measures_direct, axis=0).astype(np.double) if len(measures_direct) > 0 else np.array([], dtype=np.double)

        # compute the statistics
        self.overall_arc = compute_statistics(measures_arc)
        self.overall_direct = compute_statistics(measures_direct)

    # @TODO implement this
    # def get_indivial_arc_stats(self) -> List[Statistics]:
    #     return []

    def get_arc_stats(self) -> Statistics:
        return self.overall_arc

    def get_direct_stats(self) -> Statistics:
        return self.overall_direct

    def get_row_data(self) -> List[object]:
        return self.overall_arc.get_row_data()

    def __str__(self):
        return str(self.overall_arc)


def get_slit_locations(threshed: np.ndarray) -> np.ndarray:
    """ Gets moments (centers) of all the slits in the image

    Args:
        threshed (np.ndarray): threshed binary image expected as type of uint8

    Returns:
        np.ndarray: list of points (int32) shaped as (N, 2) with 2 being (x, y)
    """

    # get outermost contours (we don't care about small holes in slits)
    contours, _ = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # construct point list
    points = np.zeros((len(contours), 2), dtype=np.int32)

    # move through each blob and get its moments
    for ind, cont in enumerate(contours):
        M = cv2.moments(cont)

        # get centroid of contour
        center_x = round(M["m10"] / M["m00"])
        center_y = round(M["m01"] / M["m00"])

        # add point
        points[ind, :] = (center_x, center_y)
    
    return points


def get_bgr(color_map, val: (int, float), max_val: (int, float)) -> tuple:
    """ Gets a bgr color from a color map """
    d = color_map(float(val) / float(max_val))
    return (int(d[2] * 255), int(d[1] * 255), int(d[0] * 255))   # convert RGB to BGR


def contiguous_uint8(image: np.ndarray, thresh=1) -> np.ndarray:
    """ Converts an image of possible values between 0-255 to a binary contigous uint8 image

    Args:
        image (np.ndarray): input image of possible multiple types
        thresh (int): number to thresh over

    Returns:
        np.ndarray: new C-contigous array
    """
    return np.ascontiguousarray((image >= thresh), dtype=np.uint8)
