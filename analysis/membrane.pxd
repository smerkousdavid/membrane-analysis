# distutils: language=c++
# cython: cdivision=True
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False

from libcpp.vector cimport vector
from libcpp.pair cimport pair
from treescan cimport location, location_pair, Segment, Skeleton
from types cimport bool_t, uint8_t, uint32_t, int32_t, NPBOOL_t, NPUINT_t, NPINT32_t, NPUINT32_t, NPLONGLONG_t, NPFLOAT_t
cimport numpy as np


cdef extern from 'src/membrane.cpp':
    pass


cdef extern from 'include/membrane.hpp' namespace 'membrane':
    cdef cppclass Membrane:
        Skeleton membrane
        Segment diameter

        Membrane() except +
        Membrane(Skeleton &membrane) except +
        bool_t is_empty()
        long double get_distance()
        location get_start()
        location get_end()
        vector[location] get_points()
        uint32_t get_num_points()
        pair[bool_t, vector[location_pair]] get_membrane_widths(const uint8_t* image, const uint32_t rows, const uint32_t cols, const long double density, const uint32_t min_measure) nogil
