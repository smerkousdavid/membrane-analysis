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

    // simple struct to compare two scanned segments to make sure it has not already been scanned
    struct seg_hash_triple {
        LOC_t middle_left, middle_middle, middle_right;
        int num_points;
        // inline bool operator==(const seg_hash &other) const {
        //     return (abs(num_points - other.num_points) <= 1) && // make sure they're only 1 point off of each other
        //         (
        //             ( // compare the starts and the ends (easy check after the num points)
        //                 ((first == other.first) && (last == other.last)) ||
        //                 ((last == other.first) && (first == other.last))
        //             ) &&
        //             ( // their middles are aligned (this is a total of 9*2 = 18 integer checks for each hash comparison)
        //                 (middle_left == other.middle_left) || 
        //                 (middle_left == other.middle_middle) ||
        //                 (middle_left == other.middle_right) || 
        //                 (middle_middle == other.middle_left) ||
        //                 (middle_middle == other.middle_middle) ||
        //                 (middle_middle == other.middle_right) ||
        //                 (middle_right == other.middle_left) ||
        //                 (middle_right == other.middle_middle) ||
        //                 (middle_right == other.middle_right)
        //             )
        //         );
        // }
    };

    class Segment {
        public:
            LOC_VEC_t points;
            LOC_SET_t ignore_points;
            Segment();
            Segment(LOC_t initial_loc);
            ~Segment();
            int temporary_id;
            uint32_t num_points;
            long double distance;
            size_t id;
            inline bool operator==(const Segment &other) const;
            bool equals(Segment *other);
            bool not_equals(Segment *other);
            Segment copy();
            void copy_to(Segment *other);
            void add_point(LOC_t point);
            void add_distance(long double dist);
            void set_ignore(LOC_SET_t points);
            long double get_distance();
            bool is_empty();
            LOC_VEC_t get_points();
            LOC_VEC_t get_points_reversed();
            void reverse_points();
            LOC_t get_point(size_t pos);
            LOC_t get_first();
            LOC_t get_last();
            size_t get_id();
            LOC_t get_opposite_loc(LOC_t loc);
            bool is_first_or_last(LOC_t point1);
            long double get_end_distance(LOC_t loc);
            bool is_close_to(LOC_t loc);
            bool is_close_to_dist(LOC_t loc, double max_dist);
            void extend_segment_close_to(Segment *other, LOC_t close);
            void extend_segment(Segment &other);
            seg_hash_triple make_hash_triple();
            bool is_in_seg_set(const seg_hash_triple &triple, LOC_SET_t &locs);
            bool add_if_not_in_seg_set(LOC_SET_t &locs);
        private:
            void construct();
    };

    class SegmentHash {
        public:
            std::size_t operator() (const Segment &seg) const;
    };

    // segment descriptors for fast segment comparisons
    // struct branch_data {
        
    //     LOC_t first, last;
    //     uint8_t expect_in;

    //     inline bool operator==(const branch_data &other) const {
    //         return loc == other.loc;
    //     }

    //     inline bool operator!=(const branch_data &other) const {
    //         return loc != other.loc;
    //     }
    // };

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
            bool contains();
            bool is_close_to_branch(LOC_t &location, double max_dist);
            std::pair<bool, LOC_t> get_closest_branch(LOC_t &location);
            bool is_empty();
            bool follow_long_path(LOC_t start_loc, Segment *scan, Segment *full, std::unordered_set<size_t> &segs_used, LOC_SET_t &branches_used); //LOC_SET_t &ends_used);
            Segment get_diameter();
        private:
            void construct();
    };

    long double loc_distance(const LOC_t p1, const LOC_t p2);
    double low_loc_distance(const LOC_t p1, const LOC_t p2);
    
    // a simple structure to keep track of branch data when scanning the skeleton
    // struct branch_data {
    //     LOC_t loc;
    //     uint8_t expect_in;

    //     inline bool operator==(const branch_data &other) const {
    //         return loc == other.loc;
    //     }

    //     inline bool operator!=(const branch_data &other) const {
    //         return loc != other.loc;
    //     }
    // };

    std::vector<Skeleton*> search_skeleton(const uint8_t* image, const uint32_t* endpoints, const int rows, const int cols,  const int num_endpoints);
}

// namespace std {
//     template<>
//     struct hash<skeleton::branch_data> { // custom hash function for x, y location struct (max size for x or y is 2^16)
//         std::size_t operator() (const skeleton::branch_data &branch) const {
//             return std::hash<cmp::location>()(branch.loc);
//         }
//     };
// }

// add definition for a segment vector and unordered set
#define SEG_VEC_t std::vector<skeleton::Segment>
#define SEG_SET_t std::unordered_set<skeleton::Segment, skeleton::SegmentHash>

#endif