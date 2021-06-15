from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set
from analysis.types cimport bool_t, uint8_t, uint32_t, int32_t, NPBOOL_t, NPUINT_t, NPINT32_t, NPUINT32_t, NPLONGLONG_t, NPFLOAT_t
cimport numpy as np


cdef extern from 'src/treescan.cpp':
    pass


cdef extern from 'include/compare.hpp' namespace 'cmp':
    cdef struct location:
        int32_t row
        int32_t col
    
    cdef struct location_pair:
        location one
        location two
        long double distance


cdef extern from 'include/treescan.hpp' namespace 'skeleton':
    cdef cppclass Segment:
        Segment() except +
        Segment(location initial_loc) except +
        vector[location] points
        unordered_set[location] ignore_points
        long double distance
        bool_t is_empty()
    
    cdef cppclass Skeleton:
        Skeleton() except +
        vector[Segment] segments
        unordered_set[location] branch_points
        unordered_set[location] end_points
        unsigned int num_segments
        void add_segment(const Segment)
        Segment get_diameter() nogil
        bool_t is_empty()
    
    vector[Skeleton*] search_skeleton(const uint8_t* image, const uint32_t* endpoints, const int rows, const int cols, const int end_points) nogil