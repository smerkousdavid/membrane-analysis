#include <vector>
#include <utility>
#include <unordered_set>
#include <cstdint>
#include "treescan.hpp"
#include "compare.hpp"

#ifndef MEMBRANE_H
#define MEMBRANE_H

namespace membrane {

    class Membrane {
        public:
            skeleton::Skeleton membrane;
            skeleton::Segment diameter;
            LOC_VEC_t division_points;
            bool is_empty();
            long double get_distance();
            LOC_t get_start();
            LOC_t get_end();
            LOC_VEC_t get_points();
            void set_division_points(LOC_VEC_t points);
            LOC_VEC_t get_division_points();
            LOC_t get_movement_direction(const double &angle, const uint32_t step);
            inline double angle2(const double y_diff, const double x_diff);
            inline double fix_angle(const double &angle);
            std::pair<bool, double> get_angle_at(const uint32_t location, const uint32_t padding);
            std::pair<bool, cmp::location_pair> make_width_measure_at(const uint8_t* mask, const uint32_t rows, const uint32_t cols, const uint32_t location, const uint32_t padding);
            std::pair<bool, LOC_PAIR_VEC_t> get_membrane_widths(const uint8_t* mask, const uint32_t rows, const uint32_t cols, const long double density, const uint32_t min_measure);
            uint32_t get_num_points();

            Membrane();
            Membrane(skeleton::Skeleton &membrane);
            ~Membrane();
        private:
            std::pair<bool, cmp::location_pair> path_at_angle(const uint8_t* mask, const uint32_t rows, const uint32_t cols, const uint32_t location, const double angle, const uint32_t padding);
            void construct();
    };

    /*class SegmentHash {
        public:
            std::size_t operator() (const Segment &seg) const;
    };*/
}

#endif