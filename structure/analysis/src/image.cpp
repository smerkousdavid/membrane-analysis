#include <cstdint>
#include <cmath>
#include <iostream> // UNCOMMENT FOR PRINT DEBUGGING
#include <algorithm>
#include "../include/treescan.hpp"
#include "../include/hitmiss.hpp"
#include "../include/membrane.hpp"
#include "../include/statistics.hpp"
#include "../include/image.hpp"

#define INVALID_ROW -1

namespace image_process {
    bool is_row_above_value(IMAGE_DEF, const uint32_t row, const uint8_t value) {
        // scan through each column in the image to figure out if it's above this value
        for (int col = 0; col < cols; col++) {
            if (IMAGE_LOC(image, row, col, cols) < value) {
                return false;
            }
        }

        return true;
    }

    int count_row_above_value(IMAGE_DEF, const uint32_t row, const uint8_t value) {
        // scan through each column in the image to figure out if it's above this value
        int count = 0;
        for (int col = 0; col < cols; col++) {
            if (IMAGE_LOC(image, row, col, cols) >= value) {
                count++;
            }
        }

        return count;
    }

    int get_first_row_above_value(IMAGE_DEF, const uint8_t value, const int start_row) {
        for (int row = start_row; row < rows; row++) {
            if (is_row_above_value(image, rows, cols, row, value)) {
                return row;
            }
        }

        return INVALID_ROW;
    }

    int get_first_consecutive_row_above_value(IMAGE_DEF, const uint8_t value, const int consecutive, const double percent_consecutive_above) {
        int start = get_first_row_above_value(image, rows, cols, value, 0);
        
        // none were found so no possible consecutives can be found
        if (start == INVALID_ROW || percent_consecutive_above <= 0.0 || percent_consecutive_above > 1.0) return INVALID_ROW;
        const double dcols = static_cast<double>(cols);

        // keep scanning until end of image
        while (start < rows) {
            // we already have 1 scanned row above the value
            bool passed = true;
            int count = 1; // we'll use this later for our second scan start
            for (count = 1; count < consecutive; count++) {
                int count_above = count_row_above_value(image, rows, cols, start + count, value);
                if ((static_cast<double>(count_above) / cols) < percent_consecutive_above) {
                    passed = false;  // we didn't meet the threshold requirement
                    break;  // continue scanning later on
                }
            }

            // make sure all consecutive rows are valid then return the first row we scanned
            if (passed) return start;

            // if we didn't find the consecutive amount of rows and returned anything then keep scanning by shifting to the next row
            start = get_first_row_above_value(image, rows, cols, value, start + count);
            if (start == INVALID_ROW) break; // not valid
        }

        return INVALID_ROW; // by default nothing was found
    }
}