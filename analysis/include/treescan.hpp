#include <vector>
#include <utility>
#include <unordered_set>
#include <cstdint>
#include "../include/compare.hpp"

#ifndef TREESCAN_H
#define TREESCAN_H

#define LOC_t cmp::location
#define LOC_VEC_t std::vector<cmp::location>
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
            double distance;
            unsigned long id;
            inline bool operator== (const Segment &other) const;
            Segment copy();
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
            Skeleton();
            ~Skeleton();
            void add_segment(const Segment);
        private:
            void construct();
    };

    std::vector<Skeleton> search_skeleton(const uint8_t* image, const uint32_t* endpoints, const int rows, const int cols,  const int num_endpoints);
}

// add definition for a segment vector and unordered set
#define SEG_VEC_t std::vector<skeleton::Segment>
#define SEG_SET_t std::unordered_set<skeleton::Segment, skeleton::SegmentHash>

#endif