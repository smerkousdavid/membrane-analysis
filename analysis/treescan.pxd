from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set
cimport numpy as np


cdef extern from 'src/treescan.cpp':
    pass


# define types (ctypedefs are iffy)
ctypedef unsigned char uint8_t
ctypedef unsigned int uint32_t
ctypedef signed int int32_t


cdef extern from 'include/compare.hpp' namespace 'cmp':
    cdef struct location:
        int32_t row
        int32_t col


cdef extern from 'include/treescan.hpp' namespace 'skeleton':
    cdef cppclass Segment:
        Segment() except +
        Segment(location initial_loc) except +
        vector[location] points
        unordered_set[location] ignore_points
        double distance
    
    cdef cppclass Skeleton:
        Skeleton() except +
        vector[Segment] segments
        unordered_set[location] branch_points
        unordered_set[location] end_points
        unsigned int num_segments
        void add_segment(const Segment)
    
    vector[Skeleton] search_skeleton(const uint8_t* image, const uint32_t* endpoints, const int rows, const int cols, const int end_points)