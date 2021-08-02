#include <cstdint>
#include <cmath>
#include <iostream> // UNCOMMENT FOR PRINT DEBUGGING
#include <algorithm>
#include "../include/treescan.hpp"
#include "../include/hitmiss.hpp"
#include "../include/membrane.hpp"
#include "../include/statistics.hpp"

#define PI 3.1415926535897932384
#define FULL_ROT (2.0 * PI) // complete rotation
#define TANGENT_R(ANGLE) static_cast<double>(ANGLE) + (PI / 2.0)  // right tangent
#define TANGENT_L(ANGLE) static_cast<double>(ANGLE) - (PI / 2.0)  // left tangent
#define FLIP_ANGLE(ANGLE) static_cast<double>(ANGLE) + PI // flip a 180
#define TANGENT(ANGLE) TANGENT_R(ANGLE) // default tangent will just be right tangent
#define MAX_DIST_PX_DIVISOR sqrt(2)  // amount to divide max distance by (diag) 
// #define OVERLAP_CHECKS 1000 // check the past 5 measurements for overlap

namespace membrane {
    inline double fix_angle(const double &angle) {
        double angle_c = angle;
        while (angle_c < 0.0) {
            angle_c = angle_c + FULL_ROT;
        }
        while (angle_c > FULL_ROT) {
            angle_c = angle_c - FULL_ROT;
        }
        return angle_c;
    }

    inline double angle2(const double y_diff, const double x_diff) {
        const double angle = atan2(y_diff, x_diff);
        return fix_angle(angle);
    }

    Membrane::Membrane() {
        this->construct();
    }

    Membrane::Membrane(skeleton::Skeleton &membrane) {
        this->membrane = membrane;
        this->diameter = membrane.get_diameter(); // use reference membrane as we can pre-emptively cache it for other membrane objects
        this->construct();
    }

    Membrane::~Membrane() {
        // todo
    }

    void Membrane::construct() {
        // todo
    }

    bool Membrane::is_empty() {
        // if (this->membrane.is_empty()) {
        //     return true;
        // }
        
        // now check diameter
        return this->diameter.is_empty();
    }

    double Membrane::get_distance() {
        if (this->is_empty()) {
            return 0.0;
        }

        // we just care about the long segments
        return this->diameter.get_distance();
    }

    LOC_t Membrane::get_start() {
        if (this->is_empty()) {
            LOC_t n;
            n.row = -1;
            n.col = -1;
            return n;
        }

        return this->diameter.get_first();
    }

    LOC_t Membrane::get_end() {
        if (this->is_empty()) {
            LOC_t n;
            n.row = -1;
            n.col = -1;
            return n;
        }

        return this->diameter.get_last();
    }

    LOC_VEC_t Membrane::get_points() {
        if (this->is_empty()) {
            LOC_VEC_t empty;
            return empty;
        }

        return this->diameter.get_points();
    }

    uint32_t Membrane::get_num_points() {
        if (this->is_empty()) {
            return 0U;
        }

        return (uint32_t) this->diameter.points.size();
    }

    void Membrane::set_division_points(LOC_VEC_t div_points) {
        this->division_points = div_points;
    }

    LOC_VEC_t Membrane::get_division_points() {
        return this->division_points;
    }

    std::pair<bool, double> Membrane::get_angle_at(const uint32_t location, const uint32_t padding) {
        uint32_t max_points = this->get_num_points();
        // acount for one less point
        if (max_points > 0) {
            max_points--;
        }

        if (location > max_points) {
            return std::pair<bool, double>(false, -1.0); // not a valid measure
        }

        // get the average at the end
        double total = 0.0;
        uint32_t num = 0U;
        uint32_t start_ind = (location < padding) ? 0 : location - padding; // if location is before padding we're uint so needs to be 0
        uint32_t end_ind = ((location + padding) > max_points) ? max_points : location + padding; // don't go past bounds of image

        // let's not make two of the same measurement
        if (start_ind == end_ind) {
            if (start_ind > 0) {
                start_ind--; // let's just sub one
            } else if (end_ind < max_points) {
                end_ind++; // shift up one
            } else {
                return std::pair<bool, double>(false, -1.0); // 1 pixel measurements are not allowed
            }
        }

        // shift through each location and get the angles
        // scan padding before and after
        // for (int i = start_ind + 1; i < end_ind; i++) {
        LOC_t &n_cur = this->diameter.points.at(start_ind);
        LOC_t &loc_cur = this->diameter.points.at(end_ind);

        // compare location to make sure average angle is correct
        const double x_diff = static_cast<double>(loc_cur.col - n_cur.col);
        const double y_diff = static_cast<double>(n_cur.row - loc_cur.row);

        // invalid measures as we had no padded measures
        if (x_diff == 0 && y_diff == 0) {
            return std::pair<bool, double>(false, -1.0); // not a valid measure
        }

        // add the measurement
        total = angle2(y_diff, x_diff);
            // num++;  // add to the measurement counter
        // }

        /*
        for (int i = (location + 1); i < end_ind; i++) {
            LOC_t &n_cur = this->diameter.points.at(i);
            const double x_diff = static_cast<double>(loc_cur.col - n_cur.col);
            const double y_diff = static_cast<double>(loc_cur.row - n_cur.row);

            // add the measurement
            total = total + this->angle2(y_diff, x_diff);
            num++;  // add to the measurement counter
        }*/

        // valid measurement! let's return the results
        return std::pair<bool, double>(true, total);  // fix_angle(total / static_cast<double>(num)));
    }

    LOC_t Membrane::get_movement_direction(const double &angle, const uint32_t step) {
        // converts a radian to a step in x, y (up is negative)
        // [TL TT TR]
        // [ML MM MR]
        // [BR BM BR]
        // respective angles from MM are (in pi/8 units)
        // [(5-7)   (3-5)   (1-3)]
        // [(7-9)    NA     (15-1)]
        // [(9-11) (11-13)  (13-15)]

        // we'll go through the order of counterclockwise from 0
        // na stands for new angle (we'll simplify them to base 10)
        // we also correct for angles that are either out of range or negative (so we just do a full flip)
        
        LOC_t move;
        long double x_r = cos(angle);
        long double y_r = -1.0 * sin(angle);
        move.col = static_cast<int32_t>(round(static_cast<long double>(step) * x_r));
        move.row = static_cast<int32_t>(round(static_cast<long double>(step) * y_r));
        return move;

        /* OLD METHOD - not used with the new dense algorithm
        const double na = static_cast<double>((angle * 8.0) / PI);  // convert to range of 0-16
        LOC_t move;
        if (na > 15 && na <= 1) {
            move.col = 1;
            move.row = 0;
        } else if (na > 1 && na <= 3) {
            move.col = 1;
            move.row = -1;
        } else if (na > 3 && na <= 5) {
            move.col = 0;
            move.row = -1;
        } else if (na > 5 && na <= 7) {
            move.col = -1;
            move.row = -1;
        } else if (na > 7 && na <= 9) {
            move.col = -1;
            move.row = 0;
        } else if (na > 9 && na <= 11) {
            move.col = -1;
            move.row = 1;
        } else if (na > 11 && na <= 13) {
            move.col = 0;
            move.row = 1;
        } else {
            move.col = 1;
            move.row = 1;
        }

        // return the new movement
        return move;*/
    }

    std::pair<bool, cmp::location_pair> Membrane::path_at_angle(const uint8_t* mask, const uint8_t* secondary_mask, bool has_secondary, bool secondary_is_inner, const uint32_t rows, const uint32_t cols, const uint32_t location, const double angle, const uint32_t padding, const uint32_t max_secondary_scan) {
        // assuming indexing has already been done (why it's private)

        const LOC_t start = this->diameter.points.at(location); // starting location
        LOC_t cur_loc = start;
        uint32_t moves = 0U;
        double follow_angle = angle;

        // continue this path until we reach a black pixel or the end of the image
        while (true) {
            cur_loc = start + this->get_movement_direction(angle, moves);

            // we're out of bounds!
            if (cur_loc.col < 0 || cur_loc.row < 0 || cur_loc.row >= rows || cur_loc.col >= cols) {
                break;
            }

            // check image type
            if (IMAGE_LOC(mask, cur_loc.row, cur_loc.col, cols) == 0U) {  // we've hit a black pixel! we're done
                // remove one move before end
                if (moves > 0) {
                    cur_loc = start + this->get_movement_direction(angle, moves - 1);
                }
                break;
            }

            moves++; // number of moves made
        }

        // keep scanning for edge mask (scan out and in)
        if (has_secondary) {
            // @TODO implement secondary
            // uint32_t secondary_moves = 0U;
            // LOC_t secondary_loc = start;
            // while (true) {
            //     // we're out of bounds!
            //     if (cur_loc.col < 0 || cur_loc.row < 0 || cur_loc.row >= rows || cur_loc.col >= cols) {
            //         break;
            //     }

            //     if (IMAGE_LOC(secondary_mask, ))
            // }
        }

        // we've made at least 1 move and aren't at our starting location
        if (moves > 0 && cur_loc != start) {
            cmp::location_pair measure_pair;
            measure_pair.one = start; // KEEP THIS PATTERN OF one = start and two = end result
            measure_pair.two = cur_loc;
            measure_pair.distance = skeleton::loc_distance(start, cur_loc);

            // valid measurement
            return std::pair<bool, cmp::location_pair>(true, measure_pair);
        }

        // not valid
        cmp::location_pair invalid;
        return std::pair<bool, cmp::location_pair>(false, invalid);
    }

    std::pair<bool, membrane_width> Membrane::make_width_measure_at(const uint8_t* mask, const uint8_t* secondary_mask, bool has_secondary, bool secondary_is_inner,
            const uint32_t rows, const uint32_t cols, const uint32_t location, const uint32_t padding, const uint32_t max_secondary_scan) {
        if (location >= this->get_num_points() || padding == 0U) {
            membrane_width loc;
            return std::pair<bool, membrane_width>(false, loc);
        }

        // measure angle and make sure it's valid
        std::pair<bool, double> angle_measure = this->get_angle_at(location, padding);
        // LOC_t ok = this->diameter.points.at(location);
        // std::cout << "AT X " << ok.col << " Y " << ok.row << " GOOD " << angle_measure.first << " ANGLE " << angle_measure.second << std::endl;
        
        // this was an invalid measurement
        if (!angle_measure.first) {
            membrane_width loc;
            return std::pair<bool, membrane_width>(false, loc);
        }

        // a valid measurement
        const double angle = angle_measure.second;
        const double follow1 = fix_angle(TANGENT(angle)); // follow tangent line
        const double follow2 = fix_angle(FLIP_ANGLE(follow1)); // follow the opposite tangent line (other direction)

        // by default follow one side
        double side_follow1 = true;

        // let's measure both of the paths (we have two paths as the membrane is a skeleton of the actual image)
        std::pair<bool, cmp::location_pair> path1 = this->path_at_angle(mask, secondary_mask, has_secondary, secondary_is_inner, rows, cols, location, follow1, padding, max_secondary_scan);
        std::pair<bool, cmp::location_pair> path2 = this->path_at_angle(mask, secondary_mask, has_secondary, secondary_is_inner, rows, cols, location, follow2, padding, max_secondary_scan);

        // if both paths are valid let's return a new location pair that measures the entire width of the membrane
        // WARNING: KEEP THIS PATTERN OF one = start and two = end result
        membrane_width measurement;
        measurement.index = location;
        bool valid = false;
        if (path1.first && path2.first) {
            // get the inner and outer measurements
            cmp::location_pair full_width;
            full_width.one = path1.second.two;
            full_width.two = path2.second.two;
            full_width.distance = skeleton::loc_distance(full_width.one, full_width.two);

            // combine results
            measurement.inner = path1.second;
            measurement.outer = path2.second;
            measurement.distance = path1.second.distance + path2.second.distance;
            valid = true;
        } else if (path1.first) { // path1 is valid but path2 isn't
            measurement.inner = path1.second;
            measurement.distance = path1.second.distance;
            valid = true;
        } else if (path2.first) { // path2 is valid but path1 isn't
            measurement.inner = path2.second;
            measurement.distance = path2.second.distance;
            valid = true;
        }

        return std::pair<bool, membrane_width>(valid, measurement);
    }

    std::pair<bool, MEMBRANE_WIDTH_VEC_t> Membrane::get_membrane_widths(const uint8_t* mask, const uint8_t* secondary_mask, bool has_secondary, bool secondary_is_inner, 
            const uint32_t rows, const uint32_t cols, const long double density, const uint32_t min_measure, const uint32_t measure_padding, const long double max_secondary_scan_relative, const long double remove_overlap_check, const long double max_measure_diff) {
        /** density (0-1) how many of the point count should be measured, min_measure = min amount to measure */
        MEMBRANE_WIDTH_VEC_t measures;

        // nothing to measure (empty or two low/high density)
        if (this->is_empty() || density < 0 || density > 1.0) {
            return std::pair<bool, MEMBRANE_WIDTH_VEC_t>(false, measures);
        }
        
        const uint32_t num_points = this->get_num_points();

        // we need at least 3 for anything reasonable
        if (num_points < 3) {
            return std::pair<bool, MEMBRANE_WIDTH_VEC_t>(false, measures);
        }

        // spacing between points to scan width for
        uint32_t density_points = static_cast<uint32_t>((static_cast<long double>(num_points) * density));
        uint32_t back_check_points = 0U;
        const bool remove_overlap = remove_overlap_check > 0.0;

        // calculate the max amount of points to scan back
        if (remove_overlap) {
            back_check_points = static_cast<uint32_t>((static_cast<long double>(num_points)) * remove_overlap_check);
        }

        // how far out from the edges should we scan the edge mask
        uint32_t max_secondary_scan = 0U;
        if (has_secondary) {
            max_secondary_scan = static_cast<uint32_t>((static_cast<double>((rows > cols) ? rows : cols) * max_secondary_scan_relative));
        }

        // make sure we actually have a normal number (we also need to account for padded measures)
        if (density_points < (min_measure + (2 * measure_padding))) {
            density_points = min_measure + (2 * measure_padding);
        }

        // make sure we have enough pad points
        uint32_t padding = measure_padding;
        if (density_points > num_points) {
            if (num_points < (2U * padding)) {
                padding = 1U; // we'll do everything without extra padding
            }
            density_points = 1U; // shift every pixel
        }

        // plus one for center point (we'll be measuring from here)
        uint32_t cur_point = static_cast<uint32_t>(static_cast<long double>(padding) / 3.0) + 1U;
        const uint32_t offset_moves = static_cast<uint32_t>(floor(static_cast<double>(num_points) / static_cast<double>(density_points)));

        // should we scan
        const bool make_measure_diff_scan = max_measure_diff != 0.0;

        // keep measuring until we reach the last index
        bool add_point = false; // don't do a prev scan on the first measurement
        while (cur_point < num_points) {
            std::pair<bool, membrane_width> measure = this->make_width_measure_at(mask, secondary_mask, has_secondary, secondary_is_inner, rows, cols, cur_point, padding, max_secondary_scan);

            // valid measurement!
            // we don't want to break out of the loop yet because there could be "skeleton" areas where there isn't a width
            if (measure.first) {
                bool check_back = true;  // possible back-scan check
                if (make_measure_diff_scan) {
                    // make sure one measurement isn't zero
                    if (measure.second.inner.distance == 0.0 || measure.second.outer.distance == 0.0) {
                        add_point = false;  // don't add this measurement
                        check_back = false; // don't continue our scan back
                    } else {
                        // let's do a ratio of max inner outer scans
                        const long double max = static_cast<long double>(std::max(std::abs(measure.second.inner.distance), std::abs(measure.second.outer.distance)));
                        const long double min = static_cast<long double>(std::min(std::abs(measure.second.inner.distance), std::abs(measure.second.outer.distance)));

                        // make sure that ratio is within our max scan ratio
                        if ((max / min) > max_measure_diff) {
                            add_point = false;  // don't add this measurement
                            check_back = false; // don't continue our scan back
                        } // else our measurement is within a safe ratio
                    }
                } 

                // we want to continue the back-check
                if (check_back) {
                    if (remove_overlap && back_check_points > 0) {  // check if there is overlap between this and the previous measurement
                        bool continue_scan = true;
                        add_point = true; // premitive point addition

                        // keep scanning until running out of points
                        while (continue_scan) {
                            continue_scan = false; // premetive stop

                            // possible scan (if checks are in place)
                            for (int check = measures.size() - 1; check >= std::max(0, static_cast<int>(measures.size() - back_check_points)); check--) {
                                membrane_width &prev_width = measures.at(check);

                                // see if the two measurements intersect and pick smaller one if that's the case
                                if (hitmiss::do_loc_interest(prev_width.inner.two, prev_width.outer.two, measure.second.inner.two, measure.second.outer.two)) {
                                    if (prev_width.distance > measure.second.distance) {
                                        // remove the previous element
                                        measures.erase(measures.begin() + check);

                                        // go further back to make sure other points aren't longer
                                        continue_scan = true;
                                    } else { // current measurement is longer so let's quit the loop (no point in adding)
                                        add_point = false; // don't add the current measurement
                                        continue_scan = false;
                                        break;
                                    }
                                }
                            }
                        }
                    } else { // we can't do a previous scan (let's just add the measurement)
                        add_point = true;
                    }
                }

                // add the point if we passed our scans
                if (add_point) {
                    measures.push_back(measure.second);
                }
            }

            // keep measuring by the offset of the current density point
            cur_point += offset_moves;
        }

        return std::pair<bool, MEMBRANE_WIDTH_VEC_t>(true, measures);
    }

    std::pair<bool, uint32_t> Membrane::closest_index_to_point(const cmp::location point) {
        // make sure it's not empty
        if (this->is_empty()) {
            return std::pair<bool, uint32_t>(false, 0U);
        }
        
        // gets the closest index to the specified point
        std::vector<cmp::location>::iterator loc_it = this->diameter.points.begin();
        double shortest = std::abs(skeleton::low_loc_distance(*loc_it, point)); // make first measurement
        uint32_t shortest_index = 0U;
        loc_it++; // start at next location so we don't have to compare shortest everytime

        // continue scanning the entire membrane
        uint32_t cur_ind = 0U;
        while (loc_it != this->diameter.points.end()) {
            const double dist = std::abs(skeleton::low_loc_distance(*loc_it, point));
            if (dist < shortest) {
                shortest = dist; // update the shortest distance
                shortest_index = cur_ind; // update the shortest index
            }

            cur_ind++;
            loc_it++;
        }

        return std::pair<bool, uint32_t>(true, shortest_index);
    }

    std::pair<bool, double> Membrane::closest_distance_to_point(const cmp::location point) {
        // make sure it's not empty
        if (this->is_empty()) {
            return std::pair<bool, double>(false, 0.0);
        }
        
        // gets the closest distance to the specified point
        std::vector<cmp::location>::iterator loc_it = this->diameter.points.begin();
        double shortest = std::abs(skeleton::low_loc_distance(*loc_it, point)); // make first measurement
        loc_it++; // start at next location so we don't have to compare shortest everytime

        // continue scanning the entire membrane
        while (loc_it != this->diameter.points.end()) {
            const double dist = std::abs(skeleton::low_loc_distance(*loc_it, point));
            if (dist < shortest) {
                shortest = dist; // update the shortest distance
            }

            loc_it++;
        }

        return std::pair<bool, double>(true, shortest);
    }

    void Membrane::set_boundary_px(const uint32_t boundary) {
        this->boundary_px = boundary;
    }

    uint32_t Membrane::get_boundary_px() {
        return this->boundary_px;
    }

    LOC_SET_t Membrane::get_matched_points(LOC_SET_t &points, const bool use_boundary_width, const uint32_t close_match_px) {
        // this is sort of a complicated function that could potentiall return multiple points within a guarantee zone

        // iterate over the membrane diameter
        LOC_SET_t point_match = LOC_SET_t();
        std::vector<cmp::location>::iterator loc_it = this->diameter.points.begin();

        // determine if we're using the boundary set by the membrane
        const uint32_t close_match = (use_boundary_width) ? this->get_boundary_px() : close_match_px;

        // continue scanning the entire membrane
        while (loc_it != this->diameter.points.end()) {
            const cmp::location loc = *loc_it;

            // iter through point sets
            LOC_SET_t::iterator p_iter = points.begin();

            // keep iterating
            while (p_iter != points.end()) {
                // if points are matching then it's on the same line so let's throw it out
                const cmp::location compare = *p_iter;
                if (loc == compare) { // technically zero distance
                    point_match.insert(compare);
                    points.erase(p_iter);
                } else {
                    const uint32_t distance = static_cast<uint32_t>(std::ceil(std::abs(skeleton::low_loc_distance(loc, compare)))); // get distance between points

                    // compare distance to see if we're within our threshold
                    if (distance <= close_match) {
                        point_match.insert(compare);
                        points.erase(p_iter);
                    }
                }

                p_iter++;
            }

            loc_it++; // shift to next loc
        }

        return point_match;
    }

    std::pair<bool, double> Membrane::arc_distance_ind(const uint32_t start, const uint32_t end) {
        // scans from start to end inclusively (from start to end point distance)
        if (this->is_empty() || start >= this->get_num_points() || end >= this->get_num_points()) {
            return std::pair<bool, double>(false, 0.0);
        }

        // let's start getting the distance between the start and end
        double distance = 0.0;

        // iter through points inbetween two sets
        if (!this->diameter.points.empty()) {
            LOC_VEC_t::iterator start_it = this->diameter.points.begin() + start, end_it = this->diameter.points.begin() + end;
            if (start_it != end_it) {
                while ((start_it + 1) != end_it) {  // measure along the arc until we reach our end index
                    distance += skeleton::low_loc_distance(*start_it, *(start_it + 1));
                    start_it++;
                }
            }
        }

        return std::pair<bool, double>(true, distance);
    }

    std::vector<membrane_duple_measure> Membrane::make_measurements_on_membrane(LOC_SET_t &points) {
        // makes all of the duple measurements along the membrane between points in order
        std::vector<std::pair<cmp::location, uint32_t>> point_loc;

        // loop through points and get closest inds to each one... generally this requires a full membrane scan
        // @TODO let's try to figure out how to do this in one scan to make it significantly faster
        LOC_SET_t::iterator p_iter = points.begin();
        uint32_t num_points = (this->is_empty()) ? 0U : this->diameter.points.size();
        uint32_t first_index = 0U, last_index = num_points - 1;
        while (p_iter != points.end()) {
            std::pair<bool, uint32_t> closest = this->closest_index_to_point(*p_iter); 

            // valid measurement? let's save it to our loc base
            if (closest.first) {
                // let's make sure the matched index isn't the first or last index (unless if it's reallly close because usually that means the slit is out of range of the membrane)
                if (closest.second == first_index || closest.second == last_index) {
                    uint32_t next_ind;
                    cmp::location first_loc = this->diameter.get_point(static_cast<size_t>((closest.second == first_index) ? first_index : last_index));
                    bool triangular = false;

                    // let's see what data we have to work with and if a triangle comparison will work
                    if (closest.second == first_index) {
                        if (num_points > 1U) {
                            next_ind = first_index + 1U;
                            triangular = true;
                        } else {
                            triangular = false;
                        }
                    } else if (closest.second == last_index) {
                        if (num_points > 1U) {
                            next_ind = last_index - 1U;
                            triangular = true;
                        } else {
                            triangular = false;
                        }
                    }

                    // get distance to the first point to see if we're within the "range"
                    double dist_1 = skeleton::low_loc_distance(*p_iter, first_loc);
                    
                    // outside of diagonal (we might need to exclude this point)
                    if (dist_1 > 1.8) {
                        // if triangular let's first make sure we're "perpindicular" to the last point and not extending out from it
                        // we don't have to worry about the case if it's extending into the membrane because the closest_index_to_point
                        // method should have already "checked" the inner regions of the membrane for us and all those points are automatically excluded
                        
                        double passed = false;
                        if (triangular) {
                            cmp::location next_loc = this->diameter.get_point(static_cast<size_t>(next_ind));

                            // get distance to both points
                            double dist_2 = skeleton::low_loc_distance(*p_iter, next_loc);

                            // let's use the pythagorean theorem to determine if the point is "close to" being perpindicular
                            double dist_3 = skeleton::low_loc_distance(first_loc, next_loc);

                            // create the sides
                            double side_a = dist_3*dist_3;
                            double side_b_1 = dist_1*dist_1;
                            double side_b_2 = dist_2*dist_2;

                            // see if a + b_1 ~ b_2 or if a + b_2 ~ b_1 (we say roughly if the distance is within ~1.5 square units)
                            const double check_within = 2.3;
                            if ((abs((side_a + side_b_1) - side_b_2) < check_within) || (abs((side_a + side_b_2) - side_b_1) < check_within)) {
                                // we're perpindicular to the membrane edge so it's just kind of far away be still a valid point
                                // as it's not extending out the end of the membrane
                                passed = true; 
                            }
                        }

                        // if we didn't pass then let's not add this point to this membrane
                        if (!passed) {
                            p_iter++;
                            continue;
                        }
                    }
                }

                // add this point to our membrane set
                point_loc.push_back(std::pair<cmp::location, uint32_t>(*p_iter, closest.second));
            }

            p_iter++;
        }

        // now that we have the index related to each point (so let's sort by the membranes index which is the second element in the pair)
        std::sort(point_loc.begin(), point_loc.end(), [](const std::pair<cmp::location, uint32_t> &x, std::pair<cmp::location, uint32_t> &y) {
            return x.second < y.second;
        });

        // now let's construct the duple measurements by using an arc-length measure and a direct-measure
        std::vector<membrane_duple_measure> results;
        uint32_t cur_ind = 0U;

        if (!point_loc.empty()) {
            std::vector<std::pair<cmp::location, uint32_t>>::iterator p_it = point_loc.begin();
            if (p_it != point_loc.end()) {  // sanity check
                while ((p_it + 1) != point_loc.end()) {
                    std::pair<cmp::location, uint32_t> cur = *p_it, next = *(p_it + 1);

                    // make the duple measurement
                    membrane_duple_measure duple;
                    duple.start = cur.first;
                    duple.end = next.first;
                    duple.start_index = cur.second;
                    duple.end_index = next.second;

                    // get the location of the points on the membrane itself
                    if (duple.start_index >= 0 && duple.end_index < this->diameter.points.size()) {
                        duple.start_membrane = this->diameter.points.at(duple.start_index);
                        duple.end_membrane = this->diameter.points.at(duple.end_index);
                    } else { // in the awful scenario (which again should never happen) indexs are out of range... let's approximate using the locations themselves
                        duple.start_membrane = duple.start;
                        duple.end_membrane = duple.end;
                    }

                    // get the direct distance (not along the membrane) of the two closest points
                    duple.direct_distance = skeleton::low_loc_distance(duple.start_membrane, duple.end_membrane);

                    // do a successful scan
                    std::pair<bool, double> measure = this->arc_distance_ind(duple.start_index, duple.end_index);
                    if (measure.first) {
                        duple.arc_distance = measure.second;
                    } else { // should never happen but let the user know that it's bad
                        duple.arc_distance = -1;
                    }

                    // let's ignore this duple measure if the measurements are completely wrong
                    if (duple.direct_distance >= 0.9 && duple.arc_distance >= 0.9) { // less than 1 should not be possible with direct or arc distances
                        // add the measurement
                        results.push_back(duple);   
                    }

                    cur_ind++; // shift the index
                    p_it++; // shift to next one
                }
            }
        }

        return results;
    }

    std::vector<Membrane> connect_close_membranes(std::vector<Membrane> &membranes, const uint32_t padding, const long double max_angle_diff, const double max_px_from_ends) {
        std::vector<Membrane> new_membranes = membranes;
        
        // first pass will match all certain points that align really really close to the membranes (this should account for most points and remove them while scanning)
        bool connected = true;
        while (connected) {
            connected = false; // pre-emptive stop
            // std::cout << "START SCAN" << std::endl;

            std::vector<Membrane>::iterator mem_it = new_membranes.begin();
            while(mem_it != new_membranes.end()) {
                // std::cout << "USING A SCAN" << std::endl;
                Membrane mem = *mem_it;

                // get the location and angle near the end (we have to end due to vector changes to be safe (I don't feel like overdebugging some weird removal issue))
                if (mem_it->is_empty()) {
                    new_membranes.erase(mem_it);  // remove this element
                    connected = true; // let's keep scanning while removing this membrane
                    break;
                }

                // get the first last points ind, location, and angle at those locations
                const uint32_t first_ind = 0U;
                const uint32_t last_ind = mem_it->get_num_points() - 1;
                const cmp::location first_loc = mem_it->diameter.get_first();
                const cmp::location last_loc = mem_it->diameter.get_last();
                std::pair<bool, double> first_angle = mem_it->get_angle_at(first_ind + padding, padding);
                std::pair<bool, double> last_angle = mem_it->get_angle_at((padding > last_ind) ? 0 : last_ind - padding, padding);

                // make sure the angle measurements work
                if (first_angle.first && last_angle.first) {
                    const double first_angle_measure = fix_angle(FLIP_ANGLE(first_angle.second));  // first angle needs to be flipped
                    const double last_angle_measure = last_angle.second;

                    // we'll use these extended locations to check if they intersect
                    
                    const std::pair<cmp::location, cmp::location> first_extended_loc = hitmiss::rad_to_center_cartesian(first_loc, first_angle_measure, max_px_from_ends / MAX_DIST_PX_DIVISOR);
                    const std::pair<cmp::location, cmp::location> last_extended_loc = hitmiss::rad_to_center_cartesian(last_loc, last_angle_measure, max_px_from_ends / MAX_DIST_PX_DIVISOR);

                    // O(n^2) (we have to cross-scan all other membranes)
                    std::vector<Membrane>::iterator mem_in_it = new_membranes.begin();
                    while (mem_in_it != new_membranes.end()) {
                        // same deal as before (skip empty measures) they'll be removed at some point
                        if (mem_in_it->is_empty() || mem_it == mem_in_it) { // or if they're the same reference
                            mem_in_it++; // shift to next one
                            continue;
                        }

                        // let's get the measures for the second membrane
                        const uint32_t second_first_ind = 0U;
                        const uint32_t second_last_ind = mem_in_it->get_num_points() - 1;
                        const cmp::location second_first_loc = mem_in_it->diameter.get_first();
                        const cmp::location second_last_loc = mem_in_it->diameter.get_last();
                        
                        // get the distances between each pair of membrane ends
                        const double dist_first_to_first = skeleton::low_loc_distance(first_loc, second_first_loc);
                        const double dist_first_to_last = skeleton::low_loc_distance(first_loc, second_last_loc);
                        const double dist_last_to_first = skeleton::low_loc_distance(last_loc, second_first_loc);
                        const double dist_last_to_last = skeleton::low_loc_distance(last_loc, second_last_loc);

                        // get the shortest distance between the pair of membranes
                        const bool is_first_to_first = (dist_first_to_first <= max_px_from_ends) && (dist_first_to_first <= dist_first_to_last) && (dist_first_to_first <= dist_last_to_first) && (dist_first_to_first <= dist_last_to_last);
                        const bool is_first_to_last = (dist_first_to_last <= max_px_from_ends) && (dist_first_to_last <= dist_last_to_first) && (dist_first_to_last <= dist_last_to_last);
                        const bool is_last_to_first = (dist_last_to_first <= max_px_from_ends) && (dist_last_to_first <= dist_last_to_last);
                        const bool is_last_to_last = (dist_last_to_last <= max_px_from_ends);

                        // first make sure the ends are close enough (if not then let's skip measuring the ends)
                        if (!is_first_to_first && !is_first_to_last && !is_last_to_first && !is_last_to_last) {
                            mem_in_it++; // skip this membrane
                            continue;
                        }

                        // let's make a single end measurement
                        std::pair<bool, double> comparing_angle;
                        std::pair<cmp::location, cmp::location> compare_loc_pair;
                        if (is_first_to_last || is_last_to_last) {
                            comparing_angle = mem_in_it->get_angle_at((padding > second_last_ind) ? 0 : second_last_ind - padding, padding); // make measurement near last
                            
                            if (comparing_angle.first) { // create extension if it's a valid measurement
                                compare_loc_pair = hitmiss::rad_to_center_cartesian(second_last_loc, comparing_angle.second, max_px_from_ends / MAX_DIST_PX_DIVISOR);
                            }
                        } else {
                            comparing_angle = mem_in_it->get_angle_at(second_first_ind + padding, padding); // make measurement near the beginning
                            
                            if (comparing_angle.first) { // create extension if it's a valid measurement
                                comparing_angle.second = fix_angle(FLIP_ANGLE(comparing_angle.second));
                                compare_loc_pair = hitmiss::rad_to_center_cartesian(second_first_loc, comparing_angle.second, max_px_from_ends / MAX_DIST_PX_DIVISOR);
                            }
                        }

                        // make sure we were able to make the measurement
                        if (!comparing_angle.first) {
                            mem_in_it++; // skip this membrane
                            continue;
                        }

                        // make sure the end measurements are valid
                        const double compare_angle = comparing_angle.second;
                        bool matched = false; // default no match
                        if (is_first_to_first || is_first_to_last) {
                            matched = hitmiss::do_loc_interest(first_extended_loc.first, first_extended_loc.second, compare_loc_pair.first, compare_loc_pair.second) && cmp::is_within(fix_angle(FLIP_ANGLE(first_angle_measure)), compare_angle, max_angle_diff); // compare firsts to lasts measurements
                        } else if (is_last_to_first || is_last_to_last) {
                            matched = hitmiss::do_loc_interest(last_extended_loc.first, last_extended_loc.second, compare_loc_pair.first, compare_loc_pair.second) && cmp::is_within(fix_angle(FLIP_ANGLE(last_angle_measure)), compare_angle, max_angle_diff);
                        }

                        // if matched let's merge results and then break loop
                        if (matched) {
                            LOC_VEC_t first_data, second_data;

                            // our first segment will be shifted as our "second"
                            if (is_first_to_first || is_first_to_last) {
                                if (is_first_to_last) {
                                    first_data = mem_in_it->diameter.get_points();
                                } else { // is_first_to_first (we need to reverse the second dataset)
                                    first_data = mem_in_it->diameter.get_points_reversed();
                                }

                                // keep the first data the same but shifted right
                                second_data = mem_it->diameter.get_points();
                            } else {
                                if (is_last_to_first) {
                                    second_data = mem_in_it->diameter.get_points();
                                } else { // is_last_to_last (we need to reverse the second dataset)
                                    second_data = mem_in_it->diameter.get_points_reversed();
                                }

                                // keep first data the same but shifted left
                                first_data = mem_it->diameter.get_points();
                            }

                            // let's connect the two segments together with an approximate straight line
                            const LOC_t start_point = first_data.at(first_data.size() - 1); // starting location
                            const LOC_t end_point = second_data.at(0); // ending location (the beginning of the second dataset)
                            // std::cout << "CONNECT S" << start_point.col << " " << start_point.col << "  E " << end_point.col << " " << end_point.row << std::endl;
                            std::vector<cmp::location> inbetween = hitmiss::locations_between_points(start_point, end_point);
                            // std::cout << "CONNECTED S" << inbetween.at(0).col << " " << inbetween.at(0).col << "  E " << inbetween.at(inbetween.size() - 1).col << " " << inbetween.at(inbetween.size() - 1).row <<std::endl;

                            // merge results into first membrane diameter
                            if (!inbetween.empty()) {
                                first_data.insert(first_data.end(), inbetween.begin(), inbetween.end()); // copy points from inbetween data over
                                mem_it->diameter.distance += skeleton::low_loc_distance(start_point, end_point); // add the distance of the additional two points
                            }

                            if(!second_data.empty()) {
                                first_data.insert(first_data.end(), second_data.begin(), second_data.end()); // copy data from second data over
                                mem_it->diameter.distance += mem_in_it->diameter.distance; // add the distance of the second segment
                            }

                            // either way update the first set of points
                            mem_it->diameter.num_points = first_data.size();
                            mem_it->diameter.points = first_data;

                            // remove second instance
                            new_membranes.erase(mem_in_it);

                            // std::cout << "MERGED" << std::endl;
                            connected = true; // we'll break out of this loop and continue the scan
                            break;
                        }
                        mem_in_it++;

                        if (connected) break;
                    }
                } // just add this membrane because we can't make a respective measurement
                mem_it++;

                if (connected) break;
            }
        }

        return new_membranes;
    }

    std::vector<MeasureResults> measurements_along_membranes(std::vector<Membrane*> membranes, LOC_VEC_t vpoints, const bool use_boundary_width, const uint32_t close_match_px, const uint32_t max_px_from_membrane) {
        // convert the points to a set (needed for cython to behave we can't really create custom unordered sets yet)
        LOC_SET_t points = LOC_SET_t();
        LOC_VEC_t::iterator loc_it = vpoints.begin();
        while (loc_it != vpoints.end()) {
            points.insert(*loc_it);
            loc_it++;
        }

        // let's first match all of the points in our mask to their respective membranes and then we'll order them
        std::vector<LOC_SET_t> mem_points;
        // first pass will match all certain points that align really really close to the membranes (this should account for most points and remove them while scanning)
        std::vector<Membrane*>::iterator mem_it = membranes.begin();
        while(mem_it != membranes.end()) {
            Membrane *mem = *mem_it;

            // let's scan while removing points
            LOC_SET_t tmem_points = mem->get_matched_points(points, use_boundary_width, close_match_px);

            // add the mem_points to our matching vector
            mem_points.push_back(tmem_points);

            // move to next membrane
            mem_it++;
        }

        // second pass is (now with a much smaller point set) going to compare the remaining points to ALL of the membranes
        LOC_SET_t::iterator p_iter = points.begin();
        while (p_iter != points.end()) {
            const cmp::location point = *p_iter;

            // start scanning the membranes
            mem_it = membranes.begin();

            // measure first membrane
            uint32_t mem_ind = 0U;
            double shortest_dist = -1.0;  // impossible measure so it will be overwritten 

            // scan all membranes
            uint32_t cur_ind = 0U;
            bool found_dist = false; // keep a flag to make sure our scan was a success
            while(mem_it != membranes.end()) {
                // let's scan while removing points
                std::pair<bool, double> measure = (*mem_it)->closest_distance_to_point(point);

                // compare the distance
                if (measure.first) {
                    // if this the first measurement or if it's shorter (and make sure that the measurement is within our MAX distance filter)
                    if ((static_cast<uint32_t>(measure.second) < max_px_from_membrane) && (measure.second < shortest_dist || shortest_dist == -1.0)) { 
                        mem_ind = cur_ind; // set the current index
                        shortest_dist = measure.second; // update the shortest distance
                        found_dist = true; // we found a distance!
                    }
                }

                // move to next membrane
                cur_ind++;
                mem_it++;
            }

            // if we found a membrane that matched
            if (found_dist && shortest_dist != -1.0) {
                mem_points.at(mem_ind).insert(point);  // add this point to the membranes set
            }

            p_iter++;
        }

        // we're done with associating the points to their membranes. They're all out of order so it's time to sort them and make the measurements
        std::vector<MeasureResults> measurements;
        std::vector<LOC_SET_t>::iterator loc_set_it = mem_points.begin();
        mem_it = membranes.begin();

        // zipped results
        while(mem_it != membranes.end() && loc_set_it != mem_points.end()) {
            Membrane *mem = (Membrane*) *mem_it;
            LOC_SET_t pset = *loc_set_it;

            // construct the ordered results
            std::vector<membrane_duple_measure> mem_measures = mem->make_measurements_on_membrane(pset);

            // construct the distance vector
            std::vector<double> sub_measures;
            std::vector<membrane_duple_measure>::iterator m_it = mem_measures.begin();
            while (m_it != mem_measures.end()) {
                sub_measures.push_back(m_it->arc_distance);
                m_it++;
            }

            // populate results struct/class
            MeasureResults results;
            results.stats = statistics::compute_stats(sub_measures);
            results.points = pset;
            results.measures = mem_measures;
            measurements.push_back(results);

            // move to next membrane/set of points
            loc_set_it++;
            mem_it++;
        }

        // return all of the results for each membrane
        return measurements;
    }
}