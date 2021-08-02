import cv2
import numpy as np
from skimage.morphology import skeletonize
from util import *
from branches import *
# from analysis import skeleton, multi_hit_miss
from analysis.skeleton import fast_skeletonize
from analysis.hitmiss import old_get_end_points, get_image_convolve, get_branch_point_matches, old_scan_for_end, scan_for_end, scan_for_branch

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


def process_membrane(layer):
    threshed = cv2.threshold(layer, min_membrane_threshold, 255, cv2.THRESH_BINARY)[1]

    """
    dilated = cv2.dilate(threshed, np.ones((dilate_membrane, dilate_membrane), np.int8), iterations=1)
    dilated = cv2.blur(dilated, (3, 3))
    dilated = cv2.medianBlur(dilated, 7)
    dilated = cv2.threshold(dilated, 5, 255, cv2.THRESH_BINARY)[1]
    # dilated = cv2.bilateralFilter(dilated, 13, 125, 125)
    # cv2.imshow('test', dilated)
    """

    skeleton = fast_skeletonize((threshed / 255).astype(np.uint8))
    branch_points = scan_for_branch(skeleton, row_first=False)
    skeleton = skeleton.astype(np.uint8) * 255

    """
    skeleton = skeletonize(threshed / 255)
    testing = multi_hit_miss(skeleton)
    branch_points = detect_branch_points(skeleton)
    skeleton = skeleton.astype(np.uint8) * 255
    """

    # destroy the branch connections
    for x, y in branch_points:
        cv2.rectangle(skeleton, (x - 3, y - 3), (x + 3, y + 3), 0, -1)

    # subdivide each skeleton contour and find the best two edges of each contour
    contours = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]
    height, width = layer.shape[:2]
    left, top, right, bottom = 0, 0, width - 1, height - 1
    corner_lines = []
    padding = 10
    for cont in contours:
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

    """
    revised_lines = []
    # remove any overlapping or redundant connections made by the previous corner connections
    for ind, line in enumerate(overlap_lines):
        similar_lines = []
        for sub_line in overlap_lines[ind + 1:]:
            if np.isin(line, sub_line).any():
                similar_lines.append(sub_line)

        longest_line = None
        longest_length = 0
        for line in similar_lines:
            length = cv2.arcLength(line, False)
            if length > longest_length:
                longest_length = length
                longest_line = line

        revised_lines.append(line if longest_line is None else longest_line)
    """

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

    contours = cv2.findContours(overlap_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]
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

    """
    # process the dilated membrane
    contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]
    height, width = layer.shape[:2]
    left, top, right, bottom = 0, 0, width - 1, height - 1
    lines = []

    lengths = []
    # process all of the lines
    for cont in contours:
        length = cv2.arcLength(cont, False) / 2
        area = cv2.contourArea(cont, False)
        if length >= min_membrane_length and area >= min_membrane_area:
            total_length += length
            cX, cY = center_mass(cont)

            # find the start and end points
            ind = 0
            cont = cont.reshape(-1, 2)
            num = cont.shape[0]
            start = cont[0]
            s_i = 0
            e_i = num - 1
            is_middle = False
            on_edge = False
            prev_rad = []
            while ind < num - 1:
                pt1 = tuple(cont[ind])
                pt2 = tuple(cont[ind + 1])
                if pt2[0] == start[0] and pt2[1] == start[1]:
                    break

                deg = point_rad(pt1, pt2)
                if (pt2[0] - 1 <= left or pt2[0] + 1 >= right) and deg < np.pi / 4:
                    if is_middle and not on_edge and (ind - s_i) > max_membrane_width:
                        e_i = ind + 1
                        break
                    else:
                        s_i = ind
                        is_middle = True
                        on_edge = True
                elif (pt2[1] - 1 <= top or pt2[1] + 1 >= bottom) and deg > np.pi / 4:
                    if is_middle and not on_edge and (ind - s_i) > max_membrane_width:
                        e_i = ind + 1
                        break
                    else:
                        s_i = ind
                        is_middle = True
                        on_edge = True
                else:
                    on_edge = False

                # check if the last
                # if len(

                ind += 1

            smooth_contours = cv2.approxPolyDP(cont[s_i:e_i], epsilon_membrane, False).reshape(-1, 2)
            lines.append(((cX, cY), length, area, smooth_contours))

    for _, corners in corner_lines:
        for x, y in corners:
            cv2.circle(skeleton, (x, y), 7, 255, -1)
    # conts = [np.array(cont, np.int32) for cont in revised_lines]
    for line in revised_lines:
        draw_lines(skeleton, line, 100, 3)
    # cv2.drawContours(skeleton, conts, -1, 100, 3)
    ilog(skeleton, delay=0)
    # exit(0)
    """

    return revised_lines, contours, lengths


