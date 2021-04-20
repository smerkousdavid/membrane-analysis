#include <algorithm>
#include <cstdint>

#ifndef COMPARE_H
#define COMPARE_H


namespace cmp {
    struct location {
        int32_t row;
        int32_t col;

        inline bool operator==(const location &other) const {
            return row == other.row && col == other.col;
        }

        inline bool operator!=(const location &other) const {
            return row != other.row || col != other.col;
        }
    };
}

namespace std {
    template<>
    struct hash<cmp::location> { // custom hash function for x, y location struct (max size for x or y is 2^16)
        std::size_t operator() (const cmp::location &loc) const {
            return std::hash<unsigned int>()((((unsigned int)loc.row) << 16) | (((unsigned int)loc.col) && 0xFFFF));
        }
    };
}

#endif