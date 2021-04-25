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
            bool is_empty();
            long double get_distance();

            LOC_VEC_t points;
            LOC_SET_t ignore_points;
            Membrane();
            Membrane(skeleton::Skeleton membrane);
            ~Membrane();
        private:
            void construct();
    };

    /*class SegmentHash {
        public:
            std::size_t operator() (const Segment &seg) const;
    };*/
}

#endif