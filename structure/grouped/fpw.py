import math
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import os
from structure.analysis import (FootProcessWidths, average_points,
                                contiguous_uint8, get_bgr, get_slit_locations)
from structure.analysis.hitmiss import scan_for_end, get_image_convolve, scan_for_branch
from structure.analysis.membrane import (MembraneMeasureResults,
                                         measure_points_along_membrane,
                                         skeletons_to_membranes)
from structure.analysis.skeleton import fast_skeletonize
from structure.analysis.statistics import Statistics, compute_statistics
from structure.analysis.treesearch import search_image_skeleton
from structure.interactive.base import ResultLayer

# from scipy import ndimage
# from skimage.morphology import medial_axis, thin, skeletonize

DEBUG = False
DEBUG_SIZE = (800, 800)


def debug_image(name, image):
    """ Show a simple debug image """
    cv2.imshow(name, cv2.resize(image, DEBUG_SIZE, interpolation=cv2.INTER_AREA))


def make_fpw_measurements(membrane_layer: np.ndarray, slit_layer: np.ndarray, draw: bool=False, export: bool=True, settings: dict={}) -> FootProcessWidths:
    """ Makes post-processing measurements on the membrane and slit layers

    Args:
        membrane_layer (np.ndarray): layer describing the membrane edge (expected threshed)
        slit_layer (np.ndarray): layer describing the locations of the slits (expected threshed)
        draw (bool): draw the results onto an image for debugging/results
        export (bool): draw the results onto a dynamic result for exports
        settings (dict): contains settings and parameters for processing

    Returns:
        FootProcessWidths: object containing all of the statistics and results
    """
    global DEBUG

    # do some sanity checking
    if membrane_layer is None or slit_layer is None:
        raise ValueError('Membrane layer nor slit layer can be None')
    
    # check shapes between images are the same
    if membrane_layer.shape != slit_layer.shape:
        raise ValueError('Membrane layer and slit layer must have the same shape!')
    
    # check that the shapes of the images are a valid threshed image
    if (len(membrane_layer.shape) == 3 and membrane_layer.shape[-1] > 1) or (len(membrane_layer.shape) != 2 and len(membrane_layer.shape) != 3):
        raise ValueError('Membrane layer and slit layer must be in shape of (width, height, 1) or (width, height) not (width, height, N) where N > 1')

    # let's convert the images into a C-contigour uint8 (binary 0 or 1) array which is required for processing
    membrane_layer = contiguous_uint8(membrane_layer, settings.get('threshold', 1))
    slit_layer = contiguous_uint8(slit_layer, settings.get('threshold', 1))

    # apply a smoothing operator
    if settings.get('smooth', True):
        smooth_size = int(settings.get('smooth_size', 3))
        smooth_iter = int(settings.get('smooth_iterations', 5))
        # element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * smooth_size + 1, 2 * smooth_size + 1), (smooth_size, smooth_size))
        # for _ in range(smooth_iter):
        #     membrane_layer = cv2.erode(membrane_layer, element, iterations=1, borderType=cv2.BORDER_REFLECT, borderValue=0)
        #     membrane_layer = cv2.dilate(membrane_layer, element, iterations=1, borderType=cv2.BORDER_REFLECT, borderValue=0)
        for _ in range(smooth_iter):
            membrane_layer = cv2.blur(membrane_layer, (2*smooth_size + 1, 2*smooth_size + 1))

    # first we need to process the membrane layer
    membrane_skeleton = fast_skeletonize(membrane_layer)

    # debugging
    if DEBUG:
        uint8_skel = (membrane_skeleton * 255).astype(np.uint8)
        debug_image('skeleton', uint8_skel)
        # cv2.imwrite('skeleton.png', (membrane_skeleton * 255).astype(np.uint8))

    # search for the branch endpoints (hitmiss algorithm)
    end_points = scan_for_end(membrane_skeleton)

    # there are no membranes to measure against
    failure = False
    if end_points is None:
        failure = True
    elif len(end_points) == 0:
        failure = True

    if failure:
        results = FootProcessWidths()
        results.set_invalid()
        return results

    # search the entire skeleton using the referenced endpoints (returns a list of TreeSkeletons and their respective points)
    skeleton_data = search_image_skeleton(membrane_skeleton, end_points, row_first=False)

    if DEBUG:
        # matches = np.array(
        #     [
        #         [
        #             [0, 0, 1, 0, 0],
        #             [0, 0, 1, 0, 0],
        #             [0, 0, 1, 0, 0],
        #             [0, 0, 1, 0, 0],
        #             [0, 0, 1, 0, 0]
        #         ]
        #     ],
        #     dtype=np.uint8
        # )

        branch_points = scan_for_branch(membrane_skeleton)  # get_image_convolve(membrane_skeleton, matches, row_first=1, scan_edge=0)
        if len(branch_points) > 0:
            for p in branch_points:
                p = (int(p[1]), int(p[0]))
                cv2.circle(uint8_skel, p, 3, 100, -1)

            for p in end_points:
                p = (int(p[1]), int(p[0]))
                cv2.circle(uint8_skel, p, 4, 150, -1)

        debug_image('branchpoints and endpoints', uint8_skel)
        cv2.imwrite('skeleton.png', uint8_skel)
        
        # get each skeleton diameter
        if DIAMETER_ONLY:
            im_cp = np.zeros(membrane_skeleton.shape[:2] + (3,), dtype=np.uint8)
            for skel in skeleton_data:
                diameter = skel.get_diameter()
                print('branches', skel.get_branches())
                print('segs', len(skel.get_segments()))
                # print('segs', len([seg for seg in skel.get_segments() if len(seg.get_points()) > 10]))


                # draw each segment
                colors = [(50, 80, 50), (0, 255, 0), (255, 100, 10), (0, 100, 150), (255, 80, 255), (100, 20, 5), (0, 100, 100), (255, 255, 0)]
                for ind, seg in enumerate(skel.get_segments()):
                    points = seg.get_points()
                    cv2.putText(im_cp, f'{ind}', (points[0][0], points[0][1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
                    if len(points) > 0:
                        points = np.concatenate([points, points[::-1]])
                        cv2.drawContours(im_cp, [points], 0, colors[ind % len(colors)], 8)
                    else:
                        print('segment %d failed' % ind)

                points = diameter.get_points()
                if len(points) > 0:
                    points = np.concatenate([points, points[::-1]])
                    cv2.drawContours(im_cp, [points], 0, (0, 0, 255), 3)
                else:
                    print('diameter failed')

            cv2.imwrite('diameter.png', im_cp)
            debug_image('segments and diameter', im_cp)

            results = FootProcessWidths()
            # results.set_image(uint8_skel)
            # return results

    # convert skeleton data into membranes (this might take a while while searching the skeletons diameter)
    # @TODO optimize skeleton diameterization
    membranes = skeletons_to_membranes(
        skeletons=skeleton_data,
        connect_close=int(settings.get('connect_close', 1)),  # connect membranes that are nearby
        padding=int(settings.get('connect_close_padding', 10)),
        max_px_from_ends=int(settings.get('connect_close_max_px', 50)),
        max_angle_diff=((float(settings.get('connect_close_max_angle_deg', 90.0)) * math.pi) / 180.0),
        row_first=0  # return items as (col, row) aka (x, y)
    )

    # now let's get the locations of the slits along the membrane
    slit_locations = get_slit_locations(slit_layer)

    # make the point measurements using the membranes
    measures = measure_points_along_membrane(
        image=membrane_layer,  # membrane layer used to estimate membrane edge width
        membranes=membranes,
        points=slit_locations,
        max_px_from_membrane=int(settings.get('slit_max_px_from_membrane', 400)),
        density=float(settings.get('membrane_width_density', 0.1)),
        min_measure=int(settings.get('membrane_width_min_measure', 3)),
        measure_padding=int(settings.get('membrane_width_measure_padding', 10)),
        max_measure_diff=float(settings.get('membrane_width_max_measure_diff', 1.2))
    )

    # capture the results and the statistics and compile the results
    results = FootProcessWidths()
    results.set_membrane_stats(measures)
    results.compute_overall()  # compute all of the stats

    # copy all of the data over for possible later processing
    membrane_data = []
    for membrane, measure in zip(membranes, measures):
        mdata = measure.get_all_data()
        mdata.update({
            'membrane_distance': membrane.get_distance(),
            'membrane_points': membrane.get_points()
        })
        membrane_data.append(mdata)
    
    # copy the data over to the results
    results.set_data(membrane_data)

    # copy threshold image and convert to color to draw on
    if draw or export:
        color_map = plt.get_cmap(settings.get('color_map', 'inferno'))

        if draw:
            draw_on = cv2.cvtColor(membrane_layer * 230, cv2.COLOR_GRAY2BGR)
        if export:
            exp_slits = ResultLayer('fpw-slits', 'FPW Slits')
            exp_direct = ResultLayer('fpw-direct', 'FPW Direct')
            exp_arcs = ResultLayer('fpw-arcs', 'FPW Arcs')
            exp_measure = ResultLayer('fpw-measure', 'FPW Measure')

        mem_points = []
        main_ind = 0
        for membrane, measure in zip(membranes, measures):
            membrane_points = membrane.get_points()

            if export:
                mem_points.append(membrane_points.tolist())
            
            paired = measure.get_point_membrane_pairs()
            rrange = measure.get_membrane_ranges()
            arcs = measure.get_arc_distances()
            max_arc = np.amax(arcs)

            # collect the paired and ranged points
            for ind, (mem_pair, pt_range, arc) in enumerate(zip(paired, rrange, arcs)):
                # deconstruct
                l1, l2 = mem_pair
                start, end = int(pt_range[0]), int(pt_range[1])
                color = get_bgr(color_map, arc, max_arc)
                mid = average_points(tuple(l1), tuple(l2))

                if draw:
                    # draw the direct line
                    cv2.line(draw_on, tuple(l1), tuple(l2), color, settings.get('direct_width', 3))
                    
                    # draw the arc
                    segment = membrane_points[start:end]
                    segment = np.append(segment, segment[::-1], axis=0)  # so it's a closed loop
                    cv2.drawContours(draw_on, [segment], 0, color, settings.get('arc_width', 3))
                    # cv2.putText(draw_on, f'{arc}', (segment[0][0], segment[0][1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                    # print('draw ', arc)

                    # draw the label text
                    # @TODO put the putText here
                if export:
                    exp_direct.draw_line(tuple(l1), tuple(l2), exp_direct.rgb(color), settings.get('direct_width', 3), corner='round')
                    exp_arcs.draw_poly_line(exp_arcs.get_global_path('{}[{}].slice({}, {})'.format(exp_arcs.index('membranes'), main_ind, start, end)), close=False, stroke=exp_arcs.rgb(color), width=settings.get('arc_width', 3), corner='round')
                    loc = membrane_points[round((start + end) / 2.0)]
                    exp_measure.draw_text('%d' % int(arc), int(loc[0]), int(loc[1]), exp_measure.rgb(color))
            main_ind += 1

        if export:
            exp_slits.add_global_data('membranes', mem_points)

        # draw all of the slit points
        for loc in slit_locations:
            point = tuple(loc)

            if draw:
                cv2.circle(draw_on, point, settings.get('slit_radius', 3), settings.get('slit_color', (255, 166, 77)), -1)
            if export:
                exp_slits.draw_circle(point, settings.get('slit_radius', 3), exp_slits.rgb(settings.get('slit_color', (255, 166, 77))))

        # update the results image
        if draw:
            results.set_image(draw_on)
        
        if export:
            results.set_export([exp_direct, exp_arcs, exp_slits, exp_measure])
    else:
        # no image
        results.set_image(None)
        results.set_export(None)

    return results


def __single_test(path):
    global SINGLE_IMAGE
    
    if SINGLE_IMAGE:
        # path = 'C:\\Users\\smerk\\Downloads\\test.png'
        data = cv2.imread(path)
        edge = (data == [255, 255, 255]).all(axis=2).astype(np.uint8)
        slits = (data == [0, 0, 255]).all(axis=2).astype(np.uint8)
    else:
        # path = 'C:\\Users\\smerk\\Documents\\Ground Truth Project\\Fabry - 09-0598\\09-0598 blk 1-1\\out_class\\09--_4294.tiff'
        data = tifffile.imread(path)
        edge = data[-2]
        slits = data[-1]

    start = time.time()
    # skel, distance = medial_axis(edge, return_distance=True)
    # import time
    # s = time.time()
    # rows, cols = edge.shape[:2]
    # edge = cv2.UMat(edge)
    #edge = cv2.pyrUp(edge, dstsize=(2 * cols, 2 * rows))
    #for i in range(5):
    #    edge = cv2.medianBlur(edge, 7)
    # edge = edge.get()
    #edge = cv2.pyrDown(edge, dstsize=(cols, rows))

    # dist = skeletonize(edge, method='lee')
    # print(time.time() - s)

    # # Distance to the background for pixels of the skeleton
    # # dist_on_skel = distance * skel

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    # ax1.imshow(edge, cmap=plt.cm.get_cmap('gray'), interpolation='nearest')
    # ax1.axis('off')
    # ax2.imshow(dist, cmap=plt.cm.get_cmap('rainbow'), interpolation='nearest')
    # ax2.contour(edge, [0.5], colors='w')
    # ax2.axis('off')

    # fig.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0, right=1)
    # plt.show()

    res = make_fpw_measurements(edge, slits, draw=True)
    if not res.is_valid():
        cv2.imshow('invalid', edge)
        print('INVALID RESULT')
    
    print(f'Process time {time.time() - start} seconds')

    if res.has_image():
        debug_image('test', res.get_image())

    if cv2.waitKey(0) == ord('q'):
        print('exiting...')
        exit(1)
    
    print('done')

if __name__ == '__main__':
    print('Running test')
    import tifffile
    DEBUG = True  # enable debugging (showing the images)
    SINGLE_IMAGE = False  # if true then use a single image where membrane = white and slits = red if false then use a multilayer tiff fiel
    MULTIPLE = True  # use a single image or a series of images in a folder
    DIAMETER_ONLY = True  # only report the diameter results and skeleton results not any further membrane analysis

    if MULTIPLE:
        base = 'C:\\Users\\smerk\\Documents\\Ground Truth Project\\Fabry - 13-0688\\13-0688 blk 1-1\\out_class\\'  # Normal - 78-0248s\\78-0248s blk E-1\\out_class
        files = [os.path.join(base, x) for x in os.listdir(base) if os.path.isfile(os.path.join(base, x))]
        print(f'Running on {len(files)} images')
        for ind, f in enumerate(files):
            print('Testing image %d (%s)' % (ind, f))
            __single_test(f)
            print('done testin image %d' % ind)
    else:
        # path = 'C:\\Users\\smerk\\Documents\\Ground Truth Project\\Normal - 78-0248s\\78-0248s blk E-1\\out_class\\19-_5137grid.tiff'
        if SINGLE_IMAGE:
            path = 'C:\\Users\\smerk\\Downloads\\test3.png'
        else:
            # path = 'C:\\Users\\smerk\\Documents\\Ground Truth Project\\Fabry - 09-0598\\09-0598 blk 1-1\\out_class\\09--_4329.tiff' # 09--_4293.tiff 09--_4294.tiff'
            path = 'C:\\Users\\smerk\\Documents\\Ground Truth Project\\Fabry - 13-0688\\13-0688 blk 1-1\\out_class\\13-_13229.tiff' # Fabry - 09-0598\\09-0598 blk 1-3\\out_class\\09--_4294.tiff'
            # path = 'C:\\Users\\smerk\\Documents\\Ground Truth Project\\Normal - 78-0248s\\78-0248s blk E-1\\out_class\\19-_5139grid.tiff'
        __single_test(path)
    print('done')