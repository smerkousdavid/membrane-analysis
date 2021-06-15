#include <cstdint>
#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>
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

    const bool loc_on_segment(cmp::location p, cmp::location q, cmp::location r) {
        /* following 3 functions were modified from a geeksforgeeks article for line intersection
         * Given three colinear points p, q, r, the function checks if
         * point q lies on line segment 'pr'
         */
        if (q.col <= std::max(p.col, r.col) && q.col >= std::min(p.col, r.col) && q.row <= std::max(p.row, r.row) && q.row >= std::min(p.row, r.row)) {
            return true;
        }
        return false;
    }

    const uint8_t loc_orientation(cmp::location p, cmp::location q, cmp::location r) {
        /* To find orientation of ordered triplet (p, q, r).
         * The function returns following values
         * 0 --> p, q and r are colinear
         * 1 --> Clockwise
         * 2 --> Counterclockwise
         * See https://www.geeksforgeeks.org/orientation-3-ordered-points/
         * for details of below formula.
         */
        const int32_t val = (q.row - p.row) * (r.col - q.col) - (q.col - p.col) * (r.row - q.row);

        // colinear
        if (val == 0) {
            return 0U;
        }
    
        // clock or counterclock wise
        return (val > 0) ? 1U : 2U;
    }
  
    const bool do_loc_interest(cmp::location p1, cmp::location q1, cmp::location p2, cmp::location q2) {
        /* The main function that returns true if line segment 'p1q1'
         * and 'p2q2' intersect.
         */

        // general cases get the orientations
        const uint8_t o1 = loc_orientation(p1, q1, p2);
        const uint8_t o2 = loc_orientation(p1, q1, q2);
        const uint8_t o3 = loc_orientation(p2, q2, p1);
        const uint8_t o4 = loc_orientation(p2, q2, q1);
    
        // general case (if orientations don't match then they intersect)
        if (o1 != o2 && o3 != o4) {
            return true;
        }
    
        // special Cases
        // p1, q1 and p2 are colinear and p2 lies on segment p1q1
        if (o1 == 0 && loc_on_segment(p1, p2, q1)) return true;
    
        // p1, q1 and q2 are colinear and q2 lies on segment p1q1
        if (o2 == 0 && loc_on_segment(p1, q2, q1)) return true;
    
        // p2, q2 and p1 are colinear and p1 lies on segment p2q2
        if (o3 == 0 && loc_on_segment(p2, p1, q2)) return true;
    
        // p2, q2 and q1 are colinear and q1 lies on segment p2q2
        if (o4 == 0 && loc_on_segment(p2, q1, q2)) return true;
    
        // the lines don't interest
        return false;
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

    cmp::location rad_to_cartesian(cmp::location start, double radians, double radius) {
        start.col = start.col + static_cast<int32_t>(std::round(radius * std::cos(radians)));
        start.row = start.row - static_cast<int32_t>(std::round(radius * std::sin(radians)));
        return start;
    }

    std::pair<cmp::location, cmp::location> rad_to_center_cartesian(cmp::location center, double radians, double radius) {
        const int32_t diff_x = static_cast<int32_t>(std::round(radius * std::cos(radians)));
        const int32_t diff_y = static_cast<int32_t>(std::round(radius * std::sin(radians)));

        cmp::location start, end;
        start.col = center.col - diff_x;
        start.row = center.row + diff_y;
        end.col = center.col + diff_x;
        end.row = center.row - diff_y;
        return std::pair<cmp::location, cmp::location>(start, end);
    }

    std::vector<cmp::location> locations_between_points(cmp::location start, cmp::location end) {
        if (start == end) {  // there aren't any intermediate points
            return std::vector<cmp::location>();
        }

        // use Bresenham's algorithm to place discrete locations between the two points (https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm#Algorithm)
        int32_t dx = abs(end.col - start.col);
        int32_t dy = abs(start.row - end.row);
        int32_t temp;
        const int32_t sx = (end.col >= start.col) ? 1.0 : -1.0;
        const int32_t sy = (end.row >= start.row) ? 1.0 : -1.0; 
        bool swap = false;

        // if dy has greater range, then swap results
        if (dy > dx) {
            temp = dx;
            dx = dy;
            dy = temp;
            swap = true;
        }

        double error = 2*dy - dx;
        cmp::location cur = start;
        std::vector<cmp::location> locs;
        for (int i = 1; i < dx; i++) {
            if (error >= 0) {
                if (swap) {
                    cur.col = cur.col + sx;
                } else {
                    cur.row = cur.row + sy;
                    error -= 2*dx;
                }
            }
            if (swap) {
                cur.row = cur.row + sy;
            } else {
                cur.col = cur.col + sx;
                error += 2*dy;
            }

            // add the point
            locs.push_back(cur);
        }
        
        return locs;
    }
}