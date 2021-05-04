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
from membrane cimport location, location_pair, Segment, Skeleton, Membrane
from types cimport bool_t, uint8_t, uint32_t, int32_t, uint64_t, NPBOOL_t, NPUINT_t, NPINT32_t, NPUINT32_t, NPLONGLONG_t, NPFLOAT_t
from analysis.treesearch import TreeSkeleton, TreeSegment

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


cdef class LocationPair(object):
    cdef location one
    cdef location two
    cdef long double distance

    def __cinit__(self, location_pair pair):
        self.one = pair.one
        self.two = pair.two
        self.distance = pair.distance
    
    cpdef tuple get_loc(self):
        return (self.one.row, self.one.col), (self.two.row, self.two.col)

    cpdef tuple get_first_xy(self):
        return (self.one.col, self.one.row)
    
    cpdef tuple get_second_xy(self):
        return (self.two.col, self.two.row)

    cpdef tuple get_xy(self):
        return self.get_first_xy(), self.get_second_xy()

    cpdef long double get_distance(self):
        return self.distance


# create python wrapper classes for the membrane segments
cdef class SkeletonMembrane(object):
    cdef Membrane *membrane
    cdef np.ndarray points
    cdef bool_t row_first
    cdef long double distance

    def __cinit__(self, np.int32_t[:, ::1] points, long double distance):
        self.points = np.asarray(points)
        self.distance = distance

    cdef void _set(self, Membrane *membrane):
        self.membrane = membrane

    def get_points(self):
        return self.points

    cpdef list get_membrane_widths(self, np.ndarray[NPUINT_t, ndim=2, mode='c'] image, long double density, uint32_t min_measure):
        # verify that the mask image is alright
        if image is None:
            raise ValueError('Image is a null array!')
        
        if image.shape[0] == 0:
            raise ValueError('Cannot determine skeleton without image')

        cdef pair[bool_t, vector[location_pair]] found
        
        # look into why cdef func has gil static types?
        cdef uint32_t rows, cols, c_min
        cdef long double c_dens
        c_dens = density
        c_min = min_measure
        rows = <uint32_t> image.shape[0]
        cols = <uint32_t> image.shape[1]

        # with nogil:
        found = self.membrane.get_membrane_widths(&image[0, 0], rows, cols, c_dens, c_min)
        
        # construct the object list
        if not found.first:
            return None
        
        # iter through second part of pair
        cdef list result = []
        for pair in found.second:
            result.append(LocationPair(pair))

        return result

    def __len__(self):
        return len(self.points)
    
    def get_distance(self):
        return self.distance

    def __dealloc__(self):
        del self.membrane  # let's delete the skeleton backend object


cdef SkeletonMembrane make_skeleton_membrane(Skeleton ref, int row_first):
    cdef int f_ind, s_ind, point_ind
    cdef np.int32_t[:, ::1] points
    cdef unsigned int num_points
    cdef long double distance
    cdef SkeletonMembrane membrane_obj
    cdef Membrane *membrane

    # compute the skeleton diameter and have it cached
    membrane = new Membrane(ref)

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


cpdef list skeletons_to_membranes(list skeletons, int row_first=0):
    """ Converts a list of skeletons into a list of membranes

    @TODO documentation
    """
    cdef Skeleton cur
    cdef uint64_t ref

    cdef list membranes = []
    for skel in skeletons:
        ref = skel.get_c_obj_pointer()
        cur = deref(<Skeleton*> ref)
        membranes.append(make_skeleton_membrane(cur, row_first))
    return membranes


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