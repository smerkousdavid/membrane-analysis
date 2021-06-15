from typing import List
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

# local
from analysis import FootProcessWidths, contiguous_uint8, get_slit_locations, get_bgr, average_points
from interactive.base import ResultLayer
from analysis.skeleton import fast_skeletonize
from analysis.hitmiss import scan_for_end
from analysis.treesearch import search_image_skeleton
from analysis.membrane import MembraneMeasureResults, skeletons_to_membranes, measure_points_along_membrane
from analysis.statistics import Statistics, compute_statistics


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

    # first we need to process the membrane layer
    membrane_skeleton = fast_skeletonize(membrane_layer)

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
        color_map = plt.get_cmap(settings.get('color_map', 'plasma'))

        if draw:
            draw_on = cv2.cvtColor(membrane_layer * 255, cv2.COLOR_GRAY2BGR)
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

            # collect the paired and ranged points
            for ind, (mem_pair, pt_range, arc) in enumerate(zip(paired, rrange, arcs)):
                # deconstruct
                l1, l2 = mem_pair
                start, end = int(pt_range[0]), int(pt_range[1])
                color = get_bgr(color_map, ind, len(paired))
                mid = average_points(tuple(l1), tuple(l2))

                if draw:
                    # draw the direct line
                    cv2.line(draw_on, tuple(l1), tuple(l2), color, settings.get('direct_width', 3))
                    
                    # draw the arc
                    segment = membrane_points[start:end]
                    segment = np.append(segment, segment[::-1], axis=0)  # so it's a closed loop
                    cv2.drawContours(draw_on, [segment], 0, color, settings.get('arc_width', 3))

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
                cv2.circle(draw_on, point, settings.get('slit_radius', 2), settings.get('slit_color', (255, 166, 77)))
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
