# distutils: language=c++
# cython: cdivision=True
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False

from analysis.types cimport bool_t, uint8_t, uint32_t, int32_t
from libcpp.vector cimport vector

# dependencies
cdef extern from 'src/statistics.cpp':
    pass


cdef extern from 'include/statistics.hpp' namespace 'statistics':
    cdef cppclass StatsResults:
        # basic stats
        bool_t valid
        double sum
        double mean
        double variance
        double STD
        double coeff_variation
        
        # quartile stuff
        double min
        double max
        double range
        double Q1
        double median
        double Q3
        double IQR

        StatsResults() except +
        const bool_t is_valid()
        const bool_t has_quartile()
        const bool_t is_nan(const double val)

    StatsResults compute_stats(vector[double] &data)
