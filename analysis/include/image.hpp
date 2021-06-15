#include <vector>
#include <utility>
#include <unordered_set>
#include <cstdint>
#include "treescan.hpp"
#include "compare.hpp"
#include "statistics.hpp"

#ifndef IMAGE_PROCESS_H
#define IMAGE_PROCESS_H

#define IMAGE_DEF const uint8_t* image, const uint32_t rows, const uint32_t cols

namespace image_process {
    bool is_row_above_value(IMAGE_DEF, const uint32_t row, const uint8_t value);
    int get_first_row_above_value(IMAGE_DEF, const uint8_t value, const int start_row);
    int get_first_consecutive_row_above_value(IMAGE_DEF, const uint8_t value, const int consecutive, const double percent_consecutive_above);
}

#endif