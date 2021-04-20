import sys
import cv2
import numpy as np


def point_distance(pt1, pt2):
    return (((pt1[0] - pt2[0]) ** 2) + ((pt1[1] - pt2[1]) ** 2)) ** 0.5


def point_slope(pt1, pt2):
    if pt1[0] == pt2[0]:
        return sys.maxsize
    return (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])


def point_rad(pt1, pt2):
    if pt2[0] < pt1[0]:
        temp = pt1
        pt1 = pt2
        pt2 = temp
    return np.arctan(point_slope(pt1, pt2))


def perp_slope(pt1, pt2):
    slope = float(point_slope(pt1, pt2))
    if slope == 0:
        return sys.maxsize
    return -1.0 / slope


def point_center(pt1, pt2, type=int):
    return type((pt1[0] + pt2[0]) / 2), type((pt1[1] + pt2[1]) / 2)


def is_line_in_edge(cont, pt1, pt2, checks=3):
    slope = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])

    def line_to(perc):
        x = pt1[0] + (perc * (pt2[0] - pt1[0]))
        return x, slope * (x - pt1[0]) + pt1[1]

    # check if it's inside or on the edge at multiple points
    check_size = 1.0 / float(checks)
    for i in range(checks):
        if cv2.pointPolygonTest(cont, line_to(check_size * i), False) < 0:
            return False

    # all points are inside the contour
    return True


def subdivide(contours, times=1, dtype=np.float32):
    def sub(cont):
        sub_ind = 0
        if len(cont) > 1:
            sub_cont = np.zeros(((cont.shape[0] * 2) - 1, 2), dtype=dtype)
            for i in range(len(cont) - 1):
                sub_cont[sub_ind] = cont[i]
                sub_ind += 1
                sub_cont[sub_ind] = point_center(cont[i], cont[i + 1], type=float if dtype == np.float32 else int)
                sub_ind += 1
            sub_cont[sub_ind] = cont[-1]
            return sub_cont
        return cont

    sub_cont = contours.copy()
    for _ in range(times):
        sub_cont = sub(sub_cont)

    return sub_cont


def closest_to(pt, cont):
    deltas = cont - pt
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)


def center_mass(cont):
    M = cv2.moments(cont)
    return int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])


def draw_lines(img, series, color, thickness, circles=False):
    if len(series) > 0 and circles:
        cv2.circle(img, tuple(series[0]), 4, color, -1)

    for i in range(len(series) - 1):
        cv2.line(img, tuple(series[i]), tuple(series[i + 1]), color, thickness)

        if circles:
            cv2.circle(img, tuple(series[i + 1]), 4, color, -1)