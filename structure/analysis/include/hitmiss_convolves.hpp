#include <cstdint>

#define NUM_BRANCH_MATCH 19 // number of convolve branch matches
#define NUM_BRANCH_DIM 3 // dimensions of the branch hits
#define BRANCH_MATRIX_OFFSET -1  // how many rows/cols to offset in the image for the scan

// optimized list of static branches
static const uint8_t BRANCH_MATCHES[NUM_BRANCH_MATCH][NUM_BRANCH_DIM][NUM_BRANCH_DIM] = {
    {{0, 1, 0},
    {1, 1, 1},  // X
    {0, 1, 0}},

    {{1, 0, 1},
    {0, 1, 0},  // Rotated X
    {1, 0, 1}},

    {{2, 1, 2},
    {1, 1, 1},  // T
    {2, 2, 2}},

    {{1, 2, 1},
    {2, 1, 2},
    {1, 2, 2}},

    {{2, 1, 2},
    {1, 1, 2},
    {2, 1, 2}},

    {{1, 2, 2},
    {2, 1, 2},
    {1, 2, 1}},

    {{2, 2, 2},
    {1, 1, 1},
    {2, 1, 2}},

    {{2, 2, 1},
    {2, 1, 2},
    {1, 2, 1}},

    {{2, 1, 2},
    {2, 1, 1},
    {2, 1, 2}},

    {{1, 2, 1},
    {2, 1, 2},
    {2, 2, 1}},

    {{1, 0, 1},
    {0, 1, 0},  // Y
    {2, 1, 2}},

    {{0, 1, 0},
    {1, 1, 2},
    {0, 2, 1}},

    {{1, 0, 2},
    {0, 1, 1},
    {1, 0, 2}},

    {{1, 0, 2},
    {0, 1, 1},
    {1, 0, 2}},

    {{0, 2, 1},
    {1, 1, 2},
    {0, 1, 0}},

    {{2, 1, 2},
    {0, 1, 0},
    {1, 0, 1}},

    {{1, 2, 0},
    {2, 1, 1},
    {0, 1, 0}},

    {{2, 0, 1},
    {1, 1, 0},
    {2, 0, 1}},

    {{0, 1, 0},
    {2, 1, 1},
    {1, 2, 0}}
};


// optimized list to find 