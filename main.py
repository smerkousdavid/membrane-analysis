from tifffile import tifffile
from skimage.morphology import skeletonize
from util import *
# from branches import *
# from analysis import skeleton as skeletons
from analysis.skeleton import fast_skeletonize
from analysis.hitmiss import old_get_end_points, get_image_convolve, get_branch_point_matches, old_scan_for_end, scan_for_end, scan_for_branch

import cv2
import numpy as np
import glob
import sys
import os
import time
import timeit
import math as m

FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'test_imgs')


# create the membrane properties
min_membrane_threshold = 25  # the min threshold (0-255) for the membrane to be considered valid
min_membrane_length = 40  # the min membrane length
min_membrane_area = 15  # the min area of the membrane (currently not used)
max_membrane_width = 50  # the max width of the membrane (currently not used)
max_corner_edge_distance = 10  # distance for a corner to be the edge of the image
max_corner_branch_distance = 10  # distance for a corner and branch distance to be considered valid
max_corner_distance = 150  # max distance that corners can be from each other
dilate_membrane = 3
epsilon_membrane = 1
in_membrane_checks = 5


def show(img):
    cv2.imshow('Test', img)
    if cv2.waitKey(0) == ord('q'):
        exit(1)


def elem_angle(point1, point2):
    return float(np.arctan2(point2[1] - point1[1], point2[0] - point1[0]))  # swap possibly later


def angle_step(angle):
    """ converts a radian to a step in x, y (up is negative) """
    # [TL TT TR]
    # [ML MM MR]
    # [BR BM BR]
    # respective angles from MM are (in pi/8 units)
    # [(5-7)   (3-5)   (1-3)]
    # [(7-9)    NA     (15-1)]
    # [(9-11) (11-13)  (13-15)]

    # we'll go through the order of counterclockwise from 0
    na = (angle * 8.0) / np.pi  # convert to range of 0-16
    if na > 15 and na <= 1:
        return [1, 0]
    elif na > 1 and na <= 3:
        return [1, -1]
    elif na > 3 and na <= 5:
        return [0, -1]
    elif na > 5 and na <= 7:
        return [-1, -1]
    elif na > 7 and na <= 9:
        return [-1, 0]
    elif na > 9 and na <= 11:
        return [-1, 1]
    elif na > 11 and na <= 13:
        return [0, 1]
    else:
        return [1, 1]


def fix_angle(a):
    while a > (2 * np.pi):
        a -= (2 * np.pi)
    while a < 0:
        a += (2 * np.pi)
    return a


def simplify_membrane_edge(layers):
    dilated_edge = cv2.dilate(layers['edge'], np.ones((7, 7), np.uint8), iterations=3)
    merged = dilated_edge & layers['membrane']
    smooth = cv2.medianBlur(merged, 21)
    # smooth = layers['membrane']

    """
    # skeletonize edge
    skeleton = skeletonize(smooth / 255).astype(np.uint8)
    # testing = skeleton.astype(np.bool)
    # print('cython', timeit.timeit(lambda: skeletonize(smooth / 255).astype(np.uint8), number = 100))
    # print('python', timeit.timeit(lambda: skeletons.fast_skeletonize((smooth / 255).astype(np.bool)).astype(np.uint8), number = 100))
    branch_points = detect_branch_points(skeleton)
    skeleton = skeleton.astype(np.uint8) * 255
    """
    skeleton = fast_skeletonize((smooth / 255).astype(np.uint8))
    branch_points = scan_for_branch(skeleton, row_first=False)
    skeleton = skeleton.astype(np.uint8) * 255

    # cv2.imshow('skel', skeleton)
    # cv2.waitKey(0)
    # print(branch_points)


    # destroy the branch connections
    for x, y in branch_points:
        cv2.rectangle(skeleton, (x - 3, y - 3), (x + 3, y + 3), 0, -1)
    # subdivide each skeleton contour and find the best two edges of each contour
    contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    height, width = skeleton.shape[:2]
    left, top, right, bottom = 0, 0, width - 1, height - 1
    corner_lines = []
    padding = 10
    # print(contours)
    for cont in contours:
        # print(cont)
        x, y, w, h = cv2.boundingRect(cont)
        single_contour = np.zeros((h + 2 * padding, w + 2 * padding, 1), np.uint8)
        offset = np.array([(int(p_x - x + padding), int(p_y - y + padding)) for p_x, p_y in cont.reshape(-1, 2)],
                          np.int32)
        cv2.drawContours(single_contour, [offset], 0, 255, 1)
        # edges = cv2.dilate(single_contour, np.ones((5, 5), np.uint8), iterations=1)

        test_ind = 0
        corners = []
        while test_ind < 3:  # test a max of 3 branches
            passed = True
            # end_points = detect_end_points((single_contour / 255).reshape(single_contour.shape[:2]).astype(np.bool))
            # corns = cv2.goodFeaturesToTrack(edges, 2 + test_ind, 0.3, 20)
            end_points = scan_for_end((single_contour / 255).reshape(single_contour.shape[:2]).astype(np.uint8), row_first=False)

            corners = []
            if end_points is not None:
                failed = 0
                for p_x, p_y in end_points:
                    # p_x, p_y = corner.ravel()
                    p_x, p_y = int(p_x + x - padding), int(p_y + y - padding)

                    """
                    # make sure it's not a branch point
                    on_edge = True
                    for branch in branch_points:
                        if point_distance((p_x, p_y), branch) < max_corner_branch_distance:
                            on_edge = False
                            failed += 1
                            break

                    if on_edge:
                    """
                    corners.append((p_x, p_y))
                if failed == test_ind + 1:
                    passed = False

            if passed:
                break

            test_ind += 1

        # we need to have two corners to continue or we're stuck in a loop
        is_loop = len(corners) < 2

        """
        if len(corners) < 2:
            continue
        """

        vectors = cont.reshape(-1, 2)

        if is_loop:
            corner_lines.append((vectors, (vectors[0], vectors[-1])))
        else:
            first = closest_to(corners[0], vectors)
            second = closest_to(corners[1], vectors)
            start = min(first, second)
            end = max(first, second)
            corner_lines.append((vectors[start:end], (vectors[start], vectors[end])))

    # revisit the lines with corners and try to attach nearby contours
    overlap_lines = []
    ind_line = 0
    connected_lines = []
    connected_corners = []
    for line, corners in corner_lines:
        c_line = list(line.reshape(-1, 2))
        for corner in corners:
            closest = sys.maxsize
            closest_point = None
            is_end = False
            for o_line, o_corners in corner_lines[ind_line + 1:]:
                for o_corner in o_corners:
                    dist = point_distance(corner, o_corner)
                    x, y = tuple(corner)
                    if x - max_corner_edge_distance <= left or x + max_corner_edge_distance > right or \
                            y - max_corner_edge_distance <= top or y + max_corner_edge_distance >= bottom:
                        continue
                    elif dist < closest and dist < max_corner_distance:
                        closest = dist
                        closest_point = (o_line, o_corner)
                        is_end = bool(tuple(o_corner) == tuple(o_corners[-1]))

            if closest_point is not None and \
                    not (tuple(corner) in connected_corners or tuple(closest_point[1]) in connected_corners):
                s_line = list(closest_point[0].reshape(-1, 2))
                is_first_end = bool(tuple(corner) == tuple(corners[-1]))

                # order the lines properly
                if is_first_end and not is_end:
                    c_line.extend(s_line)
                elif is_first_end and is_end:
                    c_line.extend(list(reversed(s_line)))
                elif not is_first_end and not is_end:
                    c_line = list(reversed(c_line))
                    c_line.extend(s_line)
                else:
                    temp = c_line[:]
                    c_line = s_line
                    c_line.extend(temp)

                # add it to the already connected corners and lines
                connected_corners.extend([tuple(corner), tuple(closest_point[1])])
                connected_lines.append((c_line, closest_point[0]))

        # process the revised line to make sure it's up to spec
        line = np.array(c_line, np.int32)
        length = cv2.arcLength(line, False)
        if length >= min_membrane_length:
            smooth_contours = cv2.approxPolyDP(line, epsilon_membrane, False).reshape(-1, 2)
            overlap_lines.append(smooth_contours)
        ind_line += 1

    overlap_image = np.zeros((height, width), np.uint8)

    # draw the connected overlapping lines
    for line in overlap_lines:
        draw_lines(overlap_image, line, 255, 1, False)

    # draw small circles at the connected corners to smooth out the line
    for corner in connected_corners:
        cv2.circle(overlap_image, corner, 1, 255, -1)

    # determine the none connected corners
    nonconnected_corners = []
    for _, corners in corner_lines:
        for corner in corners:
            if tuple(corner) not in connected_corners:
                nonconnected_corners.append(tuple(corner))

    contours, _ = cv2.findContours(overlap_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    revised_lines = []
    lengths = []
    for cont in contours:
        vectors = cont.reshape(-1, 2)
        closest_corners = []

        for test in range(2):
            closest_corner = (0, 0)
            closest_length = sys.maxsize
            for corner in nonconnected_corners:
                if corner not in closest_corners:
                    length = point_distance(corner, vectors[closest_to(corner, vectors)])
                    if length < closest_length:
                        closest_length = length
                        closest_corner = corner
            closest_corners.append(closest_corner)

        first = closest_to(closest_corners[0], vectors)
        second = closest_to(closest_corners[1], vectors)
        start = min(first, second)
        end = max(first, second)
        line = vectors[start:end]
        revised_lines.append(line)
        lengths.append(cv2.arcLength(line, False))

    # show(overlap_image)

    # layers['membrane'] = np.zeros((500, 500), dtype=np.uint8)
    # layers['membrane'][:100, :] = 255

    # go through each line and draw tangents
    all_revised_lines = []
    all_widths = []
    for line in revised_lines:
        # let's divide the membrane into (at least let's try to 30 pixel-ish intervals)
        parts = int(max(cv2.arcLength(line, False) / 15, 2))
        divided = np.array_split(line, parts)
        widths = []

        # get two "random" points to do tangent
        # divided = [[[3, 5], [15, 4]]]
        width_lines = []
        for div in divided:
            # print(div)
            pt1 = div[0]

            # iterate for some points to get the average slope
            n = 0
            angles = []
            for i in range(1, 20):
                try:
                    angles.append(elem_angle(pt1, div[i]))
                    n += 1
                except IndexError:
                    pass
            if n <= 1:
                pt1 = div[0]
                angle = angles[0]
            else:
                pt1 = div[int(n // 2)]
                angle = np.average(angles)

            # make sure that our slope isn't ridiculous
            # if abs(slope) > height:
            #     eq = lambda step: float(pt1[1] + step)
            # else:
            # eq = lambda step: slope * (step - float(pt1[0])) + pt1[1]

            # let's create the two angles of travel (perp to current slope)
            angle_one = fix_angle(angle + (np.pi / 2.0))
            angle_two = fix_angle(angle - (np.pi / 2.0))

            # determine if length is being measured out or in as in negative x or y or
            back_count = 0
            back_line = []
            back_angle = angle_one
            back_pt = [float(pt1[0]), float(pt1[1])]
            backward_continue = True
            forward_angle = angle_two
            forward_count = 0
            forward_continue = True
            forward_line = []
            forward_pt = [float(pt1[0]), float(pt1[1])]
            subsampling = 1.0

            # make sure there is at least a few steps
            steps = max(width, height)
            steps = max(abs(steps), 5)

            # iterate until end of image but we'll break at first non-white pixel
            for i in range(steps):
                # forward_move = angle_step(forward_angle)
                if backward_continue:
                    off_x, off_y = angle_step(back_angle)
                    back_pt[0] += off_x
                    back_pt[1] += off_y

                    try:
                        if back_pt[0] > 0 and back_pt[0] < width and back_pt[1] > 0 and back_pt[1] < height and layers['membrane'][int(back_pt[1]), int(back_pt[0])] > 1:
                            back_count += 1
                            back_line.append(back_pt)
                        else:
                            backward_continue = False
                    except IndexError:
                        backward_continue = False

                if forward_continue:
                    off_x, off_y = angle_step(forward_angle)
                    forward_pt[0] += off_x
                    forward_pt[1] += off_y

                    try:
                        if forward_pt[0] > 0 and forward_pt[0] < width and forward_pt[1] > 0 and forward_pt[1] < height and layers['membrane'][int(forward_pt[1]), int(forward_pt[0])] > 1:
                            forward_count += 1
                        else:
                            forward_continue = False
                    except IndexError:
                        forward_continue = False

                # let's not continue if both loops fail
                if not backward_continue and not forward_continue:
                    break

            # let's choose forward and back methods
            forward = forward_count >= back_count
            line = forward_line + back_line
            if len(line) >= 2:
                widths.append(int(m.sqrt((line[0][0] - line[-1][0])**2 + (line[0][1] - line[-1][1])**2)))
                width_lines.append(line)
    
        # fix the lines to remove duplicates
        fixing = width_lines
        width_lines = []
        for line in fixing:
            if len(line) > 1:
                l = np.array(line, dtype=np.int32)
                width_lines.append(l)
            """
            sub_line = []
            for i in line:
                if i not in sub_line: 
                    if len(i) == 2:
                        sub_line.append(i) 

            if len(sub_line) > 1 and all([len(set(W) & set(sub_line)) == 0 for W in width_lines]):  # we're making sure no points overlap in any other line as well
                width_lines.append(sub_line)
            """

        if len(width_lines) > 1:
            # print('adding')
            all_widths.extend(widths)
            all_revised_lines.extend(width_lines)

    # print(len(all_revised_lines))
    # print(np.amax(np.array([a.flatten() for a in all_revised_lines])))
    colored_membrane = cv2.cvtColor(layers['membrane'], cv2.COLOR_GRAY2BGR)
    cv2.drawContours(colored_membrane, [np.array(line, np.int32) for line in all_revised_lines], -1, (255, 100, 0), 1)

    # draw each length
    for i in range(len(all_revised_lines)):
        cv2.putText(colored_membrane, str(all_widths[i]), (all_revised_lines[i][0][0], all_revised_lines[i][0][1]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 100, 255), 1, cv2.LINE_AA, False) 

    show(colored_membrane)
    # show(testing)

def proc_layers(image):
    return {
        'membrane': image[0],
        'edge': image[2]
    }


if __name__ == '__main__':
    files = list(glob.glob(FOLDER + os.path.sep + '*.tiff'))
    for file in files:
        name = os.path.basename(file)
        print('Testing on file %s' % name)

        # open image
        image = tifffile.imread(file)
        layers = proc_layers(image)
        simplify_membrane_edge(layers)