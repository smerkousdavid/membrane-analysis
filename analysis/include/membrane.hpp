#include <vector>
#include <utility>
#include <unordered_set>
#include <cstdint>
#include "treescan.hpp"
#include "compare.hpp"
#include "statistics.hpp"

#ifndef MEMBRANE_H
#define MEMBRANE_H

namespace membrane {

    // generally used for lines and their distances
    struct membrane_width {
        cmp::location_pair inner;
        cmp::location_pair outer;
        long double distance;
        uint32_t index;

        inline bool operator==(const membrane_width &other) const {
            return inner == other.inner && outer == other.outer;
        }

        inline bool operator!=(const membrane_width &other) const {
            return inner != other.inner || outer != other.outer;
        }

        void swap(membrane_width &other) {
            // temp copy of current params
            cmp::location_pair inner_copy = inner;
            cmp::location_pair outer_copy = outer;
            long double distance_copy = distance;
            
            // copy from other
            inner = other.inner;
            outer = other.outer;
            distance = other.distance;

            // set other from temp
            other.inner = inner_copy;
            other.outer = outer_copy;
            other.distance = distance_copy;
        }
    };

    // used for the measurement between points on a membrane
    struct membrane_duple_measure {
        cmp::location start;
        cmp::location end;
        cmp::location start_membrane;
        cmp::location end_membrane;
        uint32_t start_index;
        uint32_t end_index;
        long double arc_distance;
        long double direct_distance;

        inline bool operator==(const membrane_duple_measure &other) const {
            return start == other.start && end == other.end && start_index == other.start_index && end_index == other.end_index && start_membrane == other.start_membrane;
        }

        inline bool operator!=(const membrane_duple_measure &other) const {
            return start != other.start || end != other.end || start_index != other.start_index || end_index != other.end_index || start_membrane != other.start_membrane;
        }
    };

    class MeasureResults {
        public:
            LOC_SET_t points;
            statistics::StatsResults stats;
            std::vector<membrane_duple_measure> measures;
    };

    class Membrane {
        public:
            skeleton::Skeleton membrane;
            skeleton::Segment diameter;
            LOC_VEC_t division_points;
            uint32_t boundary_px;
            bool is_empty();
            long double get_distance();
            LOC_t get_start();
            LOC_t get_end();
            LOC_VEC_t get_points();
            void set_division_points(LOC_VEC_t points);
            LOC_VEC_t get_division_points();
            LOC_t get_movement_direction(const double &angle, const uint32_t step);
            void set_boundary_px(const uint32_t boundary);
            uint32_t get_boundary_px();
            // inline double angle2(const double y_diff, const double x_diff);
            // inline double fix_angle(const double &angle);
            double distance_to_point(const cmp::location loc);
            std::pair<bool, double> get_angle_at(const uint32_t location, const uint32_t padding);
            std::pair<bool, membrane_width> make_width_measure_at(const uint8_t* mask, const uint8_t* secondary_mask, bool has_secondary, bool secondary_is_inner, const uint32_t rows, const uint32_t cols, const uint32_t location, const uint32_t padding, const uint32_t max_secondary_scan);
            std::pair<bool, std::vector<membrane_width>> get_membrane_widths(const uint8_t* mask, const uint8_t* secondary_mask, bool has_secondary, bool secondary_is_inner, const uint32_t rows, const uint32_t cols, const long double density, const uint32_t min_measure, const uint32_t measure_padding, const long double max_secondary_scan_relative, const long double remove_overlap_check, const long double max_measure_diff);
            LOC_SET_t get_matched_points(LOC_SET_t &points, const bool use_boundary_width, const uint32_t close_match_px);
            std::pair<bool, double> closest_distance_to_point(const cmp::location point);
            std::pair<bool, uint32_t> closest_index_to_point(const cmp::location point);
            std::vector<membrane_duple_measure> make_measurements_on_membrane(LOC_SET_t &points);
            std::pair<bool, double> arc_distance_ind(const uint32_t start, const uint32_t end);
            uint32_t get_num_points();

            Membrane();
            Membrane(skeleton::Skeleton &membrane);
            ~Membrane();
        private:
            std::pair<bool, cmp::location_pair> path_at_angle(const uint8_t* mask, const uint8_t* secondary_mask, bool has_secondary, bool secondary_is_inner, const uint32_t rows, const uint32_t cols, const uint32_t location, const double angle, const uint32_t padding, const uint32_t max_secondary_scan);
            void construct();
    };

    std::vector<MeasureResults> measurements_along_membranes(std::vector<Membrane*> membranes, LOC_VEC_t vpoints, const bool use_boundary_width, const uint32_t close_match_px, const uint32_t max_px_from_membrane);

    /*class SegmentHash {
        public:
            std::size_t operator() (const Segment &seg) const;
    };*/
}

#define MEMBRANE_WIDTH_VEC_t std::vector<membrane::membrane_width>

#endif