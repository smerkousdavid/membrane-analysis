#include <cstdint>
#include <cmath>
#include <vector>
#include <iostream>
#include "../include/hitmiss.hpp"


namespace hitmiss {
    uint8_t convolve_match(const uint8_t* mat, const uint8_t* match, const uint32_t start_row, const uint32_t start_col, const uint32_t rows, const uint32_t cols) {
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                if (IMAGE_LOC(match, row + start_row, col + start_col, cols) != HIT_IDC && IMAGE_LOC(mat, row, col, cols) != IMAGE_LOC(match, row + start_row, col + start_col, cols)) {
                    return HIT_MISS;
                }
            }
        }

        return HIT_MATCH;
    }

    void get_submat_coord(const uint32_t center_row, const uint32_t center_col, const uint32_t rows, const uint32_t cols, const uint32_t match_size, int &offset_row, int &offset_col) {
        offset_row = center_row - ((match_size - 1) / 2);
        offset_col = center_col - ((match_size - 1) / 2);

        // make sure points are within image bounds
        if (offset_row < 0) {
            offset_row = 0;
        } else if (offset_row >= rows) {
            offset_row = rows - 1;
        }

        if (offset_col < 0) {
            offset_col = 0;
        } else if (offset_col >= cols) {
            offset_col = cols - 1;
        }
    }

    uint8_t convolve_match_series(const uint8_t* image, const uint8_t* matches, const uint32_t match_num,  const uint32_t start_row, const uint32_t start_col, const uint32_t rows, const uint32_t cols, const uint32_t offset_image_row, const uint32_t offset_image_col, const uint32_t image_cols) {
        for (int match = 0; match < match_num; match++) {
            for (int row = 0; row < rows; row++) {
                for (int col = 0; col < cols; col++) {
                    if (SERIES_LOC(matches, match, row + start_row, col + start_col, rows, cols) != HIT_IDC && IMAGE_LOC(image, row + offset_image_row, col + offset_image_col, image_cols) != SERIES_LOC(matches, match, row + start_row, col + start_col, rows, cols)) {
                        goto hit_missing;  // not a match
                    }
                }
            }

            // if goto statement isn't called then we return a hit
            return HIT_MATCH;

            // continue looping if missed called
            hit_missing: {}
        }

        // all goto statements were hit (so no matches)
        return HIT_MISS;
    }

    // @REMEMBER that match_size must be an odd number
    // @TODO try to support matrices that are greater than a 3x3 for now match_size must be a value of 3
    std::vector<cmp::location> convolve_match_image(const uint8_t* image, const uint8_t* matches, const uint32_t rows, const uint32_t cols, const uint32_t num_matches, const uint32_t match_size, const uint8_t scan_edge) {
        // first scan the edges as each point requires its own conditional (which we don't have to do in the middle)
        int cur_row, cur_col, last_row = rows - 1, last_col = cols - 1;
        int second_last_row = last_row - 1, second_last_col = last_col - 1;
        int offset_row, offset_col;
        std::vector<cmp::location> endpoints;
        
        // scan top row
        if (scan_edge == (uint8_t) 1) {
            cur_row = 0;
            for (int col = 0; col < second_last_col; col++) {
                if (IMAGE_LOC(image, cur_row, col, cols) == HIT_MATCH) {
                    get_submat_coord(0, col, rows, cols, match_size, offset_row, offset_col);  // get image offset location for the sub-matrix
                    if (convolve_match_series(
                        image,
                        matches,
                        num_matches,
                        1, // start row (on matches)
                        (col == 0) ? 1 : 0, // start col (on matches)
                        2, // num rows
                        (col == 0) ? 2 : 3, // num cols
                        offset_row,
                        offset_col,
                        cols
                    ) == HIT_MATCH) {
                        cmp::location loc;
                        loc.row = cur_row;
                        loc.col = col;
                        endpoints.push_back(loc);
                    }
                }
            }

            // scan right col
            cur_col = last_col;
            for (int row = 0; row < second_last_row; row++) {
                if (IMAGE_LOC(image, row, cur_col, cols) == HIT_MATCH) {
                    get_submat_coord(row, last_col, rows, cols, match_size, offset_row, offset_col);  // get image offset location for the sub-matrix
                    if (convolve_match_series(
                        image,
                        matches,
                        num_matches,
                        (row == 0) ? 1 : 0, // start row (on matches)
                        0, // start col (on matches)
                        (row == 0) ? 2 : 3, // num rows
                        2, // num cols
                        offset_row,
                        offset_col,
                        cols
                    ) == HIT_MATCH) {
                        cmp::location loc;
                        loc.row = row;
                        loc.col = cur_col;
                        endpoints.push_back(loc);
                    }
                }
            }

            // scan bottom row
            cur_row = last_row;
            for (int col = last_col; col > 0; col--) {
                if (IMAGE_LOC(image, cur_row, col, cols) == HIT_MATCH) {
                    get_submat_coord(last_row, col, rows, cols, match_size, offset_row, offset_col);  // get image offset location for the sub-matrix
                    if (convolve_match_series(
                        image,
                        matches,
                        num_matches,
                        0, // start row (on matches)
                        0, // start col (on matches)
                        2, // num rows
                        (col == last_col) ? 2 : 3,
                        offset_row,
                        offset_col,
                        cols
                    ) == HIT_MATCH) {
                        cmp::location loc;
                        loc.row = cur_row;
                        loc.col = col;
                        endpoints.push_back(loc);
                    }
                }
            }

            
            // scan left col
            cur_col = 0;
            for (int row = last_row; row > 0; row--) {
                if (IMAGE_LOC(image, row, cur_col, cols) == HIT_MATCH) {
                    get_submat_coord(row, 0, rows, cols, match_size, offset_row, offset_col);  // get image offset location for the sub-matrix
                    if (convolve_match_series(
                        image,
                        matches,
                        num_matches,
                        0, // start row (on matches)
                        1, // start col (on matches)
                        (row == last_row) ? 2 : 3, // num rows
                        2, // num cols
                        offset_row,
                        offset_col,
                        cols
                    ) == HIT_MATCH) {
                        cmp::location loc;
                        loc.row = row;
                        loc.col = cur_col;
                        endpoints.push_back(loc);
                    }
                }
            }
        }

        // scan center of image (now we don't need any bound checks for our matches or image coordinates)
        const uint32_t matrix_offset = ((match_size - 1) / 2);
        
        for (int row = 1; row < rows - 1; row++) {
            for (int col = 1; col < cols - 1; col++) {
                if (IMAGE_LOC(image, row, col, cols) == HIT_MATCH) {
                    offset_row = row - matrix_offset;
                    offset_col = col - matrix_offset;
                    // std::cout << "LOC " << row << " " << col << " " << matrix_offset << " num " << num_matches << std::endl;

                    if (convolve_match_series(image, matches, num_matches, 0, 0, match_size, match_size, offset_row, offset_col, cols)) {
                        cmp::location loc;
                        loc.row = row;
                        loc.col = col;
                        endpoints.push_back(loc);
                    }
                }
            }
        }

        return endpoints;
    }
}