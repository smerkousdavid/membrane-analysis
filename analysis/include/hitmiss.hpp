#include <vector>
#include <utility>
#include <unordered_set>
#include <cstdint>
#include "../include/compare.hpp"

#ifndef HITMISS_H
#define HITMISS_H

#define IMAGE_DTYPE (uint8_t)
#define HIT_MATCH IMAGE_DTYPE 1U  // match convolve
#define HIT_MISS IMAGE_DTYPE 0U  // miss convolve
#define HIT_IDC IMAGE_DTYPE 2U  // IDC = I don't care
#define LOC_2D(ROW, COL, COLS) (int) ((ROW) * COLS) + (COL)
#define LOC_3D(MATCH, ROW, COL, ROWS, COLS) (int) ((MATCH) * COLS * ROWS) + LOC_2D(ROW, COL, COLS)
#define IMAGE_LOC(IMAGE, ROW, COL, COLS) IMAGE_DTYPE IMAGE[LOC_2D(ROW, COL, COLS)]
#define SERIES_LOC(IMAGE, MATCH, ROW, COL, ROWS, COLS) IMAGE_DTYPE IMAGE[LOC_3D(MATCH, ROW, COL, ROWS, COLS)]


namespace hitmiss {
    std::vector<cmp::location> convolve_match_image(const uint8_t* image, const uint8_t* matches, const uint32_t rows, const uint32_t cols, const uint32_t num_matches, const uint32_t match_size);
    uint8_t convolve_match(const uint8_t* mat, const uint8_t* match, const uint32_t start_row, const uint32_t start_col, const uint32_t rows, const uint32_t cols);
    uint8_t convolve_match_series(const uint8_t* mat, const uint8_t* matches, const uint32_t match_num, const uint32_t start_row, const uint32_t start_col, const uint32_t rows, const uint32_t cols, const uint32_t offset_image_row, const uint32_t offset_image_col, const uint32_t image_cols);
    void get_submat_coord(const uint32_t center_row, const uint32_t center_col, const uint32_t rows, const uint32_t cols, const uint32_t match_size, int &offset_row, int &offset_col);
}

#endif