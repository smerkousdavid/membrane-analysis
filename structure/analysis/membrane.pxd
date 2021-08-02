# distutils: language=c++
# cython: cdivision=True
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False

from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.unordered_set cimport unordered_set
from structure.analysis.treescan cimport location, location_pair, Segment, Skeleton
from structure.analysis.statistics cimport StatsResults
from structure.analysis.types cimport bool_t, uint8_t, uint32_t, int32_t, NPBOOL_t, NPUINT_t, NPINT32_t, NPUINT32_t, NPLONGLONG_t, NPFLOAT_t
cimport numpy as np

# dependencies
cdef extern from 'src/hitmiss.cpp':
    pass

cdef extern from 'src/membrane.cpp':
    pass


cdef extern from 'include/membrane.hpp' namespace 'membrane':
    cdef struct membrane_width:
        location_pair inner
        location_pair outer
        long double distance
        uint32_t index

    cdef struct membrane_duple_measure:
        location start
        location end
        location start_membrane
        location end_membrane
        uint32_t start_index
        uint32_t end_index
        long double arc_distance
        long double direct_distance

    cdef cppclass MeasureResults:
        unordered_set[location] points
        StatsResults stats
        vector[membrane_duple_measure] measures

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
        void set_boundary_px(const uint32_t boundary)
        uint32_t get_boundary_px()
        pair[bool_t, vector[membrane_width]] get_membrane_widths(const uint8_t* mask, const uint8_t* secondary_mask, bool_t has_secondary, bool_t secondary_is_inner, const uint32_t rows, const uint32_t cols, const long double density, const uint32_t min_measure, const uint32_t measure_padding, const long double max_secondary_scan_relative, const long double remove_overlap_check, const long double max_measure_diff) nogil
    
    vector[Membrane] connect_close_membranes(vector[Membrane] &membranes, const uint32_t padding, const long double max_angle_diff, const double max_px_from_ends) nogil
    vector[MeasureResults] measurements_along_membranes(vector[Membrane*] membranes, vector[location] vpoints, const bool_t use_boundary_width, const uint32_t close_match_px, const uint32_t max_px_from_membrane) nogil
    