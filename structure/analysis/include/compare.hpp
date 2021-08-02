#include <algorithm>
#include <cstdint>
#include <cmath>

#ifndef COMPARE_H
#define COMPARE_H


namespace cmp {
    bool is_within(double val1, double val2, double error) {
        const double diff = (val1 > val2) ? val1 - val2 : val2 - val1;
        return diff <= error;
    }

    struct location {
        int32_t row;
        int32_t col;

        inline bool operator==(const location &other) const {
            return row == other.row && col == other.col;
        }

        inline bool operator!=(const location &other) const {
            return row != other.row || col != other.col;
        }

        inline location operator+(const location &other) const {
            location snew;
            snew.row = row + other.row;
            snew.col = col + other.col;
            return snew;
        }

        inline location operator-(const location &other) const {
            location snew;
            snew.row = row - other.row;
            snew.col = col - other.col;
            return snew;
        }

        inline location operator*(const location &other) const {
            location snew;
            snew.row = row * other.row;
            snew.col = col * other.col;
            return snew;
        }

        inline location operator/(const location &other) const {
            location snew;
            snew.row = row / other.row;
            snew.col = col / other.col;
            return snew;
        }
    };

    // generally used for lines and their distances
    struct location_pair {
        location one;
        location two;
        long double distance;

        inline bool operator==(const location_pair &other) const {
            return one == other.one && two == other.two;
        }

        inline bool operator!=(const location_pair &other) const {
            return one != other.one || two != other.two;
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