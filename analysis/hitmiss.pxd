from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set
from types cimport uint8_t, uint32_t, int32_t, NPBOOL_t, NPUINT_t, NPINT32_t, NPUINT32_t, NPLONGLONG_t, NPFLOAT_t
cimport numpy as np


cdef extern from 'src/hitmiss.cpp':
    pass



cdef extern from 'include/compare.hpp' namespace 'cmp':
    cdef struct location:
        int32_t row
        int32_t col


cdef extern from 'include/hitmiss.hpp' namespace 'hitmiss':
    vector[location] convolve_match_image(const uint8_t* image, const uint8_t* matches, const uint32_t rows, const uint32_t cols, const uint32_t num_matches, const uint32_t match_size, const uint8_t scan_edge)
    uint8_t convolve_match(const uint8_t* mat, const uint8_t* match, const uint32_t start_row, const uint32_t start_col, const uint32_t rows, const uint32_t cols)
    uint8_t convolve_match_series(const uint8_t* image, const uint8_t* matches, const uint32_t match_num, const uint32_t start_row, const uint32_t start_col, const uint32_t rows, const uint32_t cols, const uint32_t offset_image_row, const uint32_t offset_image_col, const uint32_t image_cols)
    void get_submat_coord(const uint32_t center_row, const uint32_t center_col, const uint32_t rows, const uint32_t cols, const uint32_t match_size, int &offset_row, int &offset_col)
