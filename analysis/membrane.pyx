# distutils: language=c++
# cython: cdivision=True
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
""" Handles image masking operations """

# cython
cimport cython
from cython.operator cimport dereference as deref
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from analysis.membrane cimport location, location_pair, membrane_width, Segment, Skeleton, Membrane, MeasureResults, membrane_duple_measure,  measurements_along_membranes, connect_close_membranes
from analysis.types cimport bool_t, uint8_t, uint32_t, int32_t, uint64_t, NPBOOL_t, NPUINT_t, NPINT32_t, NPUINT32_t, NPLONGLONG_t, NPFLOAT_t
from analysis.treesearch import TreeSkeleton, TreeSegment
from analysis.statistics import Statistics
from analysis.statistics cimport StatsResults

# numpy
cimport numpy as np
import numpy as np

# python const
NPUINT8 = np.uint8
NPFLOAT = np.float32

# cython const
cdef int dims = 3

# float color structs
cdef struct s_fcolor:
    NPFLOAT_t r, g, b


cdef class MembraneWidth(object):
    cdef location_pair inner
    cdef location_pair outer
    cdef long double distance
    cdef uint32_t index

    def __cinit__(self, membrane_width width):
        self.inner = width.inner
        self.outer = width.outer
        self.distance = width.distance
        self.index = width.index
    
    cpdef tuple get_loc(self):
        return (self.inner.two.row, self.inner.two.col), (self.outer.two.row, self.outer.two.col)

    cpdef tuple get_start_xy(self):
        return (self.inner.one.col, self.inner.one.row)

    cpdef uint32_t get_index(self):
        return self.index

    cpdef tuple get_inner_xy(self):
        return (self.inner.two.col, self.inner.two.row)
    
    cpdef tuple get_outer_xy(self):
        return (self.outer.two.col, self.outer.two.row)

    cpdef tuple get_xy(self):
        return self.get_inner_xy(), self.get_outer_xy()

    cpdef long double get_distance(self):
        return self.distance



cdef class MembraneMeasureResults(object):
    cdef np.ndarray points
    cdef np.ndarray point_pairs
    cdef np.ndarray point_membrane_pairs
    cdef np.ndarray arc_distances
    cdef np.ndarray direct_distances
    cdef np.ndarray membrane_ranges
    cdef bool_t val_is_empty
    cdef object stats

    def __init__(self):
        self.stats = None

    def get_stats(self):
        return self.stats

    def is_stats_valid(self):
        if self.stats is None:
            return False
        return self.stats.is_valid()

    cpdef np.ndarray get_points(self):
        return self.points

    cpdef np.ndarray get_point_pairs(self):
        return self.point_pairs

    cpdef np.ndarray get_point_membrane_pairs(self):
        return self.point_membrane_pairs

    cpdef np.ndarray get_arc_distances(self):
        return self.arc_distances

    cpdef np.ndarray get_direct_distances(self):
        return self.direct_distances

    cpdef np.ndarray get_membrane_ranges(self):
        return self.membrane_ranges

    cpdef bool_t is_empty(self):
        return self.val_is_empty

    def get_all_data(self):
        return {
            'points': self.points,
            'point_pairs': self.point_pairs,
            'arc_distances': self.arc_distances,
            'direct_distances': self.direct_distances,
            'membrane_ranges': self.membrane_ranges
        }

# create python wrapper classes for the membrane segments
cdef class SkeletonMembrane(object):
    cdef Membrane membrane
    cdef np.ndarray points
    cdef bool_t row_first
    cdef long double distance

    def __cinit__(self, np.int32_t[:, ::1] points, long double distance):
        self.points = np.asarray(points)
        self.distance = distance

    cdef void _set(self, Membrane membrane):
        self.membrane = membrane

    def get_points(self):
        return self.points

    cpdef list get_membrane_widths(self, np.ndarray[NPUINT_t, ndim=2, mode='c'] image, np.ndarray[NPUINT_t, ndim=2, mode='c'] secondary, long double density, uint32_t min_measure, uint32_t measure_padding, bool_t secondary_is_inner, long double edge_scan_density, long double remove_overlap_check, long double max_measure_diff):
        # verify that the mask image is alright
        if image is None:
            raise ValueError('Image is a null array!')
        
        if image.shape[0] == 0:
            raise ValueError('Cannot determine skeleton without image')

        # check secondary mask if it's valid
        if secondary is not None and secondary.shape != image.shape:
            raise ValueError('Secondary mask must have same shape as image mask')

        cdef pair[bool_t, vector[membrane_width]] found
        
        # look into why cdef func has gil static types?
        cdef uint32_t rows, cols, c_min, c_pad
        cdef long double c_dens, c_scan, c_r_overlap, c_max_diff
        cdef bool_t has_secondary, c_is_inner
        cdef uint8_t *second_mask

        # secondary image pointer
        if secondary is None:
            second_mask = NULL
            c_scan = 0.0
            has_secondary = <bool_t> 0
            c_is_inner = <bool_t> 0
        else:
            second_mask = &secondary[0, 0]
            c_scan = edge_scan_density
            has_secondary = <bool_t> 1
            c_is_inner = secondary_is_inner

        # copy c vars
        c_dens = density
        c_min = min_measure
        c_pad = measure_padding
        c_r_overlap = remove_overlap_check
        c_max_diff = max_measure_diff
        rows = <uint32_t> image.shape[0]
        cols = <uint32_t> image.shape[1]

        with nogil:
            found = self.membrane.get_membrane_widths(&image[0, 0], second_mask, has_secondary, c_is_inner, rows, cols, c_dens, c_min, c_pad, c_scan, c_r_overlap, c_max_diff)
        
        # construct the object list
        if not found.first:
            return None
        
        # iter through second part of pair
        cdef list result = []
        for width in found.second:
            result.append(MembraneWidth(width))

        return result

    def __len__(self):
        return len(self.points)
    
    def get_distance(self):
        return self.distance
    
    cpdef is_empty(self):
        cdef bool_t val_is_empty = self.membrane.is_empty()
        return val_is_empty

    cpdef uint64_t get_c_obj_pointer(self) except *:
        return <uint64_t> &self.membrane

    # def __dealloc__(self):
    #    del self.membrane  # let's delete the membrane backend object


cdef SkeletonMembrane make_skeleton_membrane(Membrane &membrane, int row_first):
    cdef int f_ind, s_ind, point_ind
    cdef np.int32_t[:, ::1] points
    cdef unsigned int num_points
    cdef long double distance
    cdef SkeletonMembrane membrane_obj
    # cdef Membrane *membrane

    # compute the skeleton diameter and have it cached
    # membrane = new Membrane(ref)

    """  UNCOMMENT TO TEST SKELETON/MEMBRANE POINT COMPARISON. TO BE REMOVED EVENTUALLY
    # simple copies
    print('IS EMPTY', ref.is_empty(), membrane.is_empty(), membrane.membrane.is_empty(), membrane.diameter.is_empty())
    cdef Segment diam = ref.get_diameter()
    num_points = diam.points.size()
    distance = diam.distance

    # reconstruct the numpy array point list (contiguous c array)
    points = np.zeros((num_points, 2), dtype=np.int32, order='c')
    # with nogil:
    # specify first and second points
    point_ind = <int> 0        
    if row_first == <int> 1:
        f_ind = <int> 0
        s_ind = <int> 1
    else:
        f_ind = <int> 1
        s_ind = <int> 0

    cdef vector[location] pp = membrane.get_points()
    points = np.zeros((pp.size(), 2), dtype=np.int32, order='c')
    print('SIZEOF', pp.size(), 'COMPARE', ref.get_diameter().points.size())
    for point in pp:
        points[point_ind, f_ind] = point.row
        points[point_ind, s_ind] = point.col
        point_ind += 1

    print('POINTS', np.asarray(points), num_points)
    """

    # make sure it's not empty before constructing it
    if membrane.is_empty():
        return None

    # simple copies
    num_points = membrane.get_num_points()
    distance = membrane.get_distance()

    # reconstruct the numpy array point list (contiguous c array) for the diameter
    points = np.zeros((num_points, 2), dtype=np.int32, order='c')
    with nogil:
        # specify first and second points
        point_ind = <int> 0        
        if row_first == <int> 1:
            f_ind = <int> 0
            s_ind = <int> 1
        else:
            f_ind = <int> 1
            s_ind = <int> 0

        for point in membrane.diameter.points:
            points[point_ind, f_ind] = point.row
            points[point_ind, s_ind] = point.col
            point_ind += 1

    # make the python object (which has some struct structure underneath which is why need to call _set)
    membrane_obj = SkeletonMembrane(points, distance)
    membrane_obj._set(membrane)
    return membrane_obj


cdef MembraneMeasureResults make_measurement_result(MeasureResults &res):
    cdef StatsResults stats = res.stats
    cdef vector[membrane_duple_measure] mes = res.measures
    cdef np.ndarray points
    cdef np.ndarray point_pairs
    cdef np.ndarray point_membrane_pairs
    cdef np.ndarray arc_distances
    cdef np.ndarray direct_distances
    cdef np.ndarray membrane_ranges
    cdef bool_t is_empty = mes.size() == 0
    cdef uint32_t cur_ind

    if is_empty:
        points = np.zeros((0, 2), np.int32)
        point_pairs = np.zeros((0, 2, 2), np.int32)
        point_membrane_pairs = np.zeros((0, 2, 2), np.int32)
        arc_distances = np.zeros((0,), np.double)
        direct_distances = np.zeros((0,), np.double)
        membrane_ranges = np.zeros((0,2), np.int32)
    else:
        points = np.zeros((mes.size() + 1, 2), np.int32)
        point_pairs = np.zeros((mes.size(), 2, 2), np.int32)
        point_membrane_pairs = np.zeros((mes.size(), 2, 2), np.int32)
        arc_distances = np.zeros((mes.size(),), np.double)
        direct_distances = np.zeros((mes.size(),), np.double)
        membrane_ranges = np.zeros((mes.size(),2), np.int32)

        # add start point (for the first location)
        points[0, :] = (mes.at(0).start.col, mes.at(0).start.row)

        # iterate through the duple and add the end points and the distances
        cur_ind = 0
        for duple in mes:
            points[cur_ind + 1, :] = (duple.end.col, duple.end.row)

            # actaul points
            point_pairs[cur_ind, 0, :] = (duple.start.col, duple.start.row)
            point_pairs[cur_ind, 1, :] = (duple.end.col, duple.end.row)

            # closest points on the membrane
            point_membrane_pairs[cur_ind, 0, :] = (duple.start_membrane.col, duple.start_membrane.row)
            point_membrane_pairs[cur_ind, 1, :] = (duple.end_membrane.col, duple.end_membrane.row)

            # distance along membrane
            arc_distances[cur_ind] = duple.arc_distance

            # direct distance from the closest points on the membrane
            direct_distances[cur_ind] = duple.direct_distance

            # the index (start, end) pairs for the membrane arcs
            membrane_ranges[cur_ind, :] = (max(duple.start_index, 0), duple.end_index)
            cur_ind += 1

    # create the statistics python object
    pystats = Statistics(<uint64_t> &stats)

    # create the python object
    cdef MembraneMeasureResults result
    result = MembraneMeasureResults()
    result.stats = pystats
    result.points = points
    result.point_pairs = point_pairs
    result.point_membrane_pairs = point_membrane_pairs
    result.arc_distances = arc_distances
    result.direct_distances = direct_distances
    result.membrane_ranges = membrane_ranges
    result.val_is_empty = is_empty

    return result


cpdef list skeletons_to_membranes(list skeletons, int connect_close=1, uint32_t padding=10, long double max_angle_diff=0.1, double max_px_from_ends=10, int row_first=0):
    """ Converts a list of skeletons into a list of membranes, by also removing empty skeletons, and connecting nearby membrane elements

    Args:
        skeletons (list of TreeSkeleton): A list of TreeSkeleton objects
        connect_close (bool): if True then connect membranes that are close to each other with the other parameters below
        padding (unsigned int): amount of pixels in the membrane to get an average angle reading out the edge of the membrane
        max_angle_diff (double): in radians how close to the two measurements have to be in order to count the two membranes as close
        max_px_from_ends (double): how many pixels/sub-pixels can the two membrane edges before to consider them "close"
        row_first (bool): should the measurements that are returned in the numpy array be (row, col) (True) or (col, row) (False: default)
    """
    cdef Skeleton cur
    cdef SkeletonMembrane mem
    cdef Membrane cur_mem
    cdef uint64_t ref

    cdef vector[Membrane] membrane_cp
    for skel in skeletons:
        if not isinstance(skel, TreeSkeleton):
            raise ValueError('All input skeletons must be of type TreeSkeleton')

        # get skeleton data from object
        ref = skel.get_c_obj_pointer()
        cur = deref(<Skeleton*> ref)

        # convert to a membrane object and copy the membrane data over
        # mem = make_skeleton_membrane(cur, row_first)
        # ref = mem.get_c_obj_pointer()
        cur_mem = Membrane(cur)  # deref(<Membrane*> ref)

        # add it to our membrane list
        membrane_cp.push_back(cur_mem)

    # connect nearby membranes using the c++ function
    if connect_close == <int> 1:
        with nogil:
            membrane_cp = connect_close_membranes(membrane_cp, padding, max_angle_diff, max_px_from_ends)

    # convert the connected membranes (most of time very few connections are found) to python objects
    cdef list membranes = []
    for cur_mem in membrane_cp:
        membranes.append(make_skeleton_membrane(cur_mem, row_first))

    return membranes


cpdef list measure_points_along_membrane(np.ndarray[NPUINT_t, ndim=2, mode='c'] image, list membranes, np.ndarray[NPINT32_t, ndim=2, mode='c'] points, uint32_t max_px_from_membrane, long double density, uint32_t min_measure, uint32_t measure_padding, long double max_measure_diff):
    if len(membranes) == 0:
        return []
    
    # for each membrane get the widths to use for our max measurement close match (this will make sure points in 3D space (next layer) are automatically matched)
    cdef uint32_t ind = 0
    cdef vector[Membrane*] membrane_refs
    cdef vector[location] measure_points
    cdef uint64_t ref
    cdef long double total, average
    cdef uint32_t count

    # iter through all membranes
    for ind in range(len(membranes)):
        # get the width measurements for the current membrane
        widths = membranes[ind].get_membrane_widths(
            image=image,  # mask of the membrane
            secondary=None,  # mask to identify which part of the membrane is the inside or the outside
            density=density,  # % of membrane p oints to scan width for
            min_measure=min_measure,  # min amount of measures for a single membrane (this won't be used if the membrane is less than N-px)
            measure_padding=measure_padding,  #  between each measurement how many points to go left/right to get an average tangent angle
            secondary_is_inner=True,  # is the secondary mask the "inner" measurement
            edge_scan_density=0.1,  # % of image dimension that we can use to scan the secondary mask from the edge of the image mask
            remove_overlap_check=1.0,  # % of membrane to scan back for possible overlaps (0-1) use 0.0 to disable overlap checks, and 1.0 to check all of them
            max_measure_diff=max_measure_diff  # max difference between the measurements from the center line of the skeleton to the edge (if one measure in or out is greater than this ratio then exclude it) use 0.0 to disable this feature
        )

        # if the widths came back as None then this is not a membrane that should be measured (so set max to be 1 pixel)
        if widths is None:
            average = 2
        elif len(widths) == 0:
            average = 2
        else:
            # get the average width
            total = 0.0
            count = 0
            for width in widths:
                total = width.get_distance()
                count = count + 1
            
            # now get the average
            average = (<long double> total) / (<long double> count)

            # make sure it's greater than 2
            if average < 2:
                average = 2

        # capture the reference pointers to add them to our membrane pointer list (so we don't have uneccessary copies)
        ref = membranes[ind].get_c_obj_pointer()
        
        # update the references boundary pixel count
        (<Membrane*> ref).set_boundary_px(<uint32_t> (average - 1))

        # add pointer to our final list
        membrane_refs.push_back(<Membrane*> ref)

    # use the calculated boundary pixels from the averages above (the references are updated for the measurements)
    cdef bool_t use_boundary_width = 1
    cdef uint32_t close_match_px = 0 

    # construct the membrane points list from the numpy array
    cdef location loc
    cdef uint32_t point_size = points.shape[0]
    ind = 0  # restart at 0
    while ind < point_size:        
        # get the row and column at the given location
        loc.col = points[ind, 0]
        loc.row = points[ind, 1]

        # add it to our vector
        measure_points.push_back(loc)

        ind += 1

    # call the c++ optimized function
    cdef vector[MeasureResults] c_res
    
    with nogil:
        c_res = measurements_along_membranes(membrane_refs, measure_points, use_boundary_width, close_match_px, max_px_from_membrane)

    # let's deconstruct the result into a list objects
    cdef list results = []
    cdef MeasureResults membrane_result

    for membrane_result in c_res:
        # make the new result object
        results.append(make_measurement_result(membrane_result))

    return results


cdef np.ndarray[NPUINT_t, ndim=3, mode='c'] blend_mask(int dlen, np.ndarray[NPUINT_t, ndim=3, mode='c'] background, np.ndarray[NPUINT_t, ndim=4, mode='c'] stack, s_fcolor *colors, NPUINT_t min_thresh, NPFLOAT_t alpha):
    """ Blends the background image with a set of binary masks
    
    :note: this will modify the background in place and not make a copy
    :param dlen: the height of the stack
    :param background: the BGR background image (np.uint8) (h, w, 3)
    :param stack: the stack of binary images (np.uint8) (d, h, w, 1)
    :param colors: the struct of colors that matches the depth of the stack
    :param min_thresh: the min class thresh to consider it a valid class identification
    :param alpha: the alpha to apply to each color
    :return: returns a blended image
    """
    assert background.dtype == NPUINT8
    assert stack.dtype == NPUINT8
    assert sizeof(colors) > 0
    assert dlen == stack.shape[0]
    assert stack.shape[1] == background.shape[0] and stack.shape[2] == background.shape[1]

    # define the width and height and background values
    cdef NPFLOAT_t cur = 0.0
    cdef NPFLOAT_t b_b = 0.0
    cdef NPFLOAT_t b_g = 0.0
    cdef NPFLOAT_t b_r = 0.0
    cdef int width = background.shape[1]
    cdef int height = background.shape[0]
    cdef np.ndarray b_copy = background.copy()
    cdef NPUINT_t[:, :, ::1] b_view = b_copy
    cdef NPUINT_t[:, :, :, ::1] s_view = stack

    # loop through each col and row to determine the pixel values (without gil lock)
    with nogil:
        for row in range(height):
            for col in range(width):
                # get the background color at that index
                b_b = <NPFLOAT_t> b_view[row, col, 0]
                b_g = <NPFLOAT_t> b_view[row, col, 1]
                b_r = <NPFLOAT_t> b_view[row, col, 2]

                # loop through the stack and calculate the new value
                for depth in range(dlen):
                    cur = <NPFLOAT_t> s_view[depth, row, col, 0]
                    if cur > min_thresh:  # if it passes the min threshold apply the new alpha blend
                        b_b = (alpha * colors[depth].b * cur) + ((1.0 - alpha) * b_b)
                        b_g = (alpha * colors[depth].g * cur) + ((1.0 - alpha) * b_g)
                        b_r = (alpha * colors[depth].r * cur) + ((1.0 - alpha) * b_r)

                # update the final color at that pixel
                b_view[row, col, 0] = <NPUINT_t> b_b
                b_view[row, col, 1] = <NPUINT_t> b_g
                b_view[row, col, 2] = <NPUINT_t> b_r

    return b_copy