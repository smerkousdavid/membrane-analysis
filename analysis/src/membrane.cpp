#include <cstdint>
#include <cmath>
#include <iostream> // UNCOMMENT FOR PRINT DEBUGGING
#include "../include/treescan.hpp"
#include "../include/hitmiss.hpp"
#include "../include/membrane.hpp"

#define PI 3.1415926535897932384
#define FULL_ROT (2.0 * PI) // complete rotation
#define TANGENT_R(ANGLE) static_cast<double>(ANGLE) + (PI / 2.0)  // right tangent
#define TANGENT_L(ANGLE) static_cast<double>(ANGLE) - (PI / 2.0)  // left tangent
#define FLIP_ANGLE(ANGLE) static_cast<double>(ANGLE) + PI // flip a 180
#define TANGENT(ANGLE) TANGENT_R(ANGLE) // default tangent will just be right tangent
#define MEASURE_PADDING 2  // measure two pixels out each direction


namespace membrane {
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
        if (this->membrane.is_empty()) {
            return true;
        }
        
        // now check diameter
        return this->diameter.is_empty();
    }

    long double Membrane::get_distance() {
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

    inline double Membrane::angle2(const double y_diff, const double x_diff) {
        const double angle = atan2(y_diff, x_diff);
        if (angle < 0.0) {
            return angle + FULL_ROT;
        } else if (angle > FULL_ROT) {
            return angle - FULL_ROT;
        }
        return angle;
    }

    inline double Membrane::fix_angle(const double &angle) {
        double angle_c = angle;
        while (angle_c < 0.0) {
            angle_c = angle_c + FULL_ROT;
        }
        while (angle_c > FULL_ROT) {
            angle_c = angle_c - FULL_ROT;
        }
        return angle_c;
    }

    std::pair<bool, double> Membrane::get_angle_at(const uint32_t location, const uint32_t padding) {
        const uint32_t max_points = this->get_num_points();
        if (location >= max_points) {
            return std::pair<bool, double>(false, -1.0); // not a valid measure
        }

        // get the average at the end
        double total = 0.0;
        uint32_t num = 0U;
        uint32_t start_ind = (location < padding) ? 0 : location - padding; // if location is before padding we're uint so needs to be 0
        uint32_t end_ind = ((location + padding) > max_points) ? max_points : location + padding; // don't go past bounds of image

        // shift through each location and get the angles
        LOC_t &loc_cur = this->diameter.points.at(start_ind);

        // scan padding before and after
        for (int i = start_ind + 1; i < end_ind; i++) {
            LOC_t &n_cur = this->diameter.points.at(i);
            const double x_diff = static_cast<double>(n_cur.col - loc_cur.col);
            const double y_diff = static_cast<double>(n_cur.row - loc_cur.row);

            // add the measurement
            total = total + this->angle2(y_diff, x_diff);
            num++;  // add to the measurement counter
        }

        /*
        for (int i = (location + 1); i < end_ind; i++) {
            LOC_t &n_cur = this->diameter.points.at(i);
            const double x_diff = static_cast<double>(loc_cur.col - n_cur.col);
            const double y_diff = static_cast<double>(loc_cur.row - n_cur.row);

            // add the measurement
            total = total + this->angle2(y_diff, x_diff);
            num++;  // add to the measurement counter
        }*/

        // invalid measures as we had no padded measures
        if (num == 0U) {
            return std::pair<bool, double>(false, -1.0); // not a valid measure
        }

        // valid measurement! let's return the results
        return std::pair<bool, double>(true, fix_angle(total / static_cast<double>(num)));
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

    std::pair<bool, cmp::location_pair> Membrane::path_at_angle(const uint8_t* mask, const uint32_t rows, const uint32_t cols, const uint32_t location, const double angle, const uint32_t padding) {
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
                break;
            }

            moves++; // number of moves made
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

    std::pair<bool, cmp::location_pair> Membrane::make_width_measure_at(const uint8_t* mask, const uint32_t rows, const uint32_t cols, const uint32_t location, const uint32_t padding) {
        if (location >= this->get_num_points()) {
            cmp::location_pair loc;
            return std::pair<bool, cmp::location_pair>(false, loc);
        }

        // measure angle and make sure it's valid
        std::pair<bool, double> angle_measure = this->get_angle_at(location, padding);
        LOC_t ok = this->diameter.points.at(location);
        std::cout << "AT X " << ok.col << " Y " << ok.row << " GOOD " << angle_measure.first << " ANGLE " << angle_measure.second << std::endl;
        
        // this was an invalid measurement
        if (!angle_measure.first) {
            cmp::location_pair loc;
            return std::pair<bool, cmp::location_pair>(false, loc);
        }

        // a valid measurement
        const double angle = angle_measure.second;
        const double follow1 = fix_angle(TANGENT(angle)); // follow tangent line
        const double follow2 = fix_angle(FLIP_ANGLE(follow1)); // follow the opposite tangent line (other direction)

        // let's measure both of the paths (we have two paths as the membrane is a skeleton of the actual image)
        std::pair<bool, cmp::location_pair> path1 = this->path_at_angle(mask, rows, cols, location, follow1, padding);
        std::pair<bool, cmp::location_pair> path2 = this->path_at_angle(mask, rows, cols, location, follow2, padding);

        // if both paths are valid let's return a new location pair that measures the entire width of the membrane
        // WARNING: KEEP THIS PATTERN OF one = start and two = end result
        if (path1.first && path2.first) {
            cmp::location_pair full_width;
            full_width.one = path1.second.two;
            full_width.two = path2.second.two;
            full_width.distance = skeleton::loc_distance(full_width.one, full_width.two);

            std::cout << "first X " << path1.second.two.col << " Y " << path1.second.two.row << std::endl;
            return std::pair<bool, cmp::location_pair>(true, full_width);
        } else if (path1.first) {
            return path1;  // path1 is valid but path2 isn't
        } else if (path2.first) {
            return path2;  // path2 is valid but path1 isn't
        }

        // none of the paths were valid (which means it was an empty skeleton line)
        cmp::location_pair invalid;
        return std::pair<bool, cmp::location_pair>(false, invalid);
    }

    std::pair<bool, LOC_PAIR_VEC_t> Membrane::get_membrane_widths(const uint8_t* mask, const uint32_t rows, const uint32_t cols, const long double density, const uint32_t min_measure) {
        /** density (0-1) how many of the point count should be measured, min_measure = min amount to measure */
        LOC_PAIR_VEC_t measures;

        // nothing to measure (empty or two low/high density)
        if (this->is_empty() || density < 0 || density > 1.0) {
            return std::pair<bool, LOC_PAIR_VEC_t>(false, measures);
        }
        
        const uint32_t num_points = this->get_num_points();

        // we need at least 3 for anything reasonable
        if (num_points < 3) {
            return std::pair<bool, LOC_PAIR_VEC_t>(false, measures);
        }

        uint32_t density_points = static_cast<uint32_t>((static_cast<long double>(num_points) * density));
        
        // make sure we actually have a normal number (we also need to account for padded measures)
        if (density_points < (min_measure + (2 * MEASURE_PADDING))) {
            density_points = min_measure + (2 * MEASURE_PADDING);
        }

        // make sure we have enough pad points
        uint32_t padding = MEASURE_PADDING;
        if (density_points > num_points) {
            if (num_points < (2 * padding)) {
                padding = 1; // we'll do everything without extra padding
            }
            density_points = 1; // shift every pixel
        }

        // plus one for center point (we'll be measuring from here)
        uint32_t cur_point = padding + 1;
        const uint32_t offset_moves = static_cast<uint32_t>(floor(static_cast<double>(num_points) / static_cast<double>(density_points)));

        // keep measuring until we reach the last index
        while (cur_point < num_points) {
            std::pair<bool, cmp::location_pair> measure = this->make_width_measure_at(mask, rows, cols, cur_point, padding);

            // valid measurement!
            // we don't want to break out of the loop yet because there could be "skeleton" areas where there isn't a width
            if (measure.first) {
                measures.push_back(measure.second);
            }

            // keep measuring by the offset of the current density point
            cur_point += offset_moves;
        }

        return std::pair<bool, LOC_PAIR_VEC_t>(true, measures);
    }

}