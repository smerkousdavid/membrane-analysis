#include <vector>
#include <utility>
#include <unordered_set>
#include <cstdint>
#include "../include/compare.hpp"

#ifndef TREESCAN_H
#define TREESCAN_H

#define LOC_t cmp::location
#define LOC_VEC_t std::vector<cmp::location>
#define LOC_PAIR_VEC_t std::vector<cmp::location_pair>
#define LOC_SET_t std::unordered_set<cmp::location>


namespace skeleton {
    class Segment {
        public:
            LOC_VEC_t points;
            LOC_SET_t ignore_points;
            Segment();
            Segment(LOC_t initial_loc);
            ~Segment();
            uint32_t num_points;
            long double distance;
            unsigned long id;
            inline bool operator== (const Segment &other) const;
            Segment copy();
            void copy_to(Segment *other);
            void add_point(LOC_t point);
            void add_distance(long double dist);
            void set_ignore(LOC_SET_t points);
            long double get_distance();
            bool is_empty();
            LOC_VEC_t get_points();
            LOC_VEC_t get_points_reversed();
            LOC_t get_point(size_t pos);
            LOC_t get_first();
            LOC_t get_last();
            LOC_t get_opposite_loc(LOC_t loc);
            bool is_first_or_last(LOC_t point1);
            long double get_end_distance(LOC_t loc);
            bool is_close_to(LOC_t loc);
            void extend_segment_close_to(Segment &other, LOC_t close);
            void extend_segment(Segment &other);
        private:
            void construct();
    };

    class SegmentHash {
        public:
            std::size_t operator() (const Segment &seg) const;
    };

    class Skeleton {
        public:
            std::vector<Segment> segments;
            LOC_SET_t branch_points;
            LOC_SET_t end_points;
            uint32_t num_segments;
            bool diameter_dirty;
            Segment diameter;
            Skeleton();
            ~Skeleton();
            void add_segment(const Segment);
            bool is_empty();
            bool follow_long_path(LOC_t start_loc, Segment &scan, Segment *full, LOC_SET_t &ends_used);
            Segment get_diameter();
        private:
            void construct();
    };

    long double loc_distance(const LOC_t p1, const LOC_t p2);
    std::vector<Skeleton*> search_skeleton(const uint8_t* image, const uint32_t* endpoints, const int rows, const int cols,  const int num_endpoints);
}

// add definition for a segment vector and unordered set
#define SEG_VEC_t std::vector<skeleton::Segment>
#define SEG_SET_t std::unordered_set<skeleton::Segment, skeleton::SegmentHash>

#endif