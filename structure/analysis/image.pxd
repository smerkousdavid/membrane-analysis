# distutils: language=c++
# cython: cdivision=True
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
from structure.analysis.types cimport bool_t, uint8_t, uint32_t, int32_t, NPBOOL_t, NPUINT_t, NPINT32_t, NPUINT32_t, NPLONGLONG_t, NPFLOAT_t
cimport numpy as np

# dependencies
cdef extern from 'src/image.cpp':
    pass

cdef extern from 'include/image.hpp' namespace 'image_process':
    int get_first_consecutive_row_above_value(const uint8_t* image, const uint32_t rows, const uint32_t cols, const uint8_t value, const int consecutive, const double percent_consecutive_above)
