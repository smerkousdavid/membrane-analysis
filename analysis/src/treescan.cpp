#include <cstdint>
#include <cmath>
#include <iostream> // UNCOMMENT FOR PRINT DEBUGGING
#include "../include/treescan.hpp"
#include "../include/hitmiss.hpp"

#define NUM_DIMS 2 // number of dimensions (row, col)
#define TL 0
#define TT 1
#define TR 2
#define ML 3
#define MR 4
#define BL 5
#define BB 6
#define BR 7
#define NONE 8  // 8 = no-move
#define POSITIONS 8  // number of total positions, that are movable, above
#define ORTHOG (double) sqrt(2)
#define NUM_BRANCH_MATCH 19 // number of convolve branch matches
#define NUM_BRANCH_DIM 3 // dimensions of the branch hits
                            // ORDER for arrays: TL, TT, TR, ML, MR, BL, BB, BR (so top left to bottom right) 
static const double DIST[POSITIONS] = {ORTHOG, 1.0, ORTHOG, 1.0, 1.0, ORTHOG, 1.0, ORTHOG};
static const signed int OFFSETS_ROW[POSITIONS] = {-1, -1, -1, 0, 0, 1, 1, 1};
static const signed int OFFSETS_COL[POSITIONS] = {-1, 0, 1, -1, 1, -1, 0, 1};
static unsigned long segment_count = 0;

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
#define BRANCH_MATRIX_OFFSET -1  // how many rows/cols to offset in the image for the scan


namespace skeleton {
    /* SEGMENT START */
    Segment::Segment() {
        this->construct();
    }

    Segment::Segment(LOC_t initial_loc) {
        this->construct();
        this->points.push_back(initial_loc);
    }

    Segment::~Segment() {}

    void Segment::construct() {
        // create unique identifier
        ++segment_count;
        this->id = segment_count; // set custom id

        this->num_points = 1;
        this->ignore_points = LOC_SET_t();
        this->distance = 0.0;
        this->points = LOC_VEC_t();
    }

    Segment Segment::copy() {
        Segment new_seg = Segment();
        new_seg.id = this->id;
        new_seg.num_points = this->num_points;
        new_seg.ignore_points = this->ignore_points;
        new_seg.distance = this->distance;
        new_seg.points = this->points;
        return new_seg;
    }

    inline bool Segment::operator== (const Segment &other) const {
        return this->id == other.id;
    }

    std::size_t SegmentHash::operator() (const Segment &seg) const {
        return std::hash<unsigned long>()(seg.id);
    }
    /* SEGMENT END */

    /* SKELETON START */
    Skeleton::Skeleton() {
        this->construct();
    }

    Skeleton::~Skeleton() {}

    void Skeleton::add_segment(const Segment seg) {
        this->segments.push_back(seg);
    }

    void Skeleton::construct() {
        this->num_segments = (uint32_t) 0;
        this->branch_points = LOC_SET_t();
        this->end_points = LOC_SET_t();
    }
    /* SKELETON END */

    std::vector<Skeleton> search_skeleton(const uint8_t* image, const uint32_t* endpoints, const int rows, const int cols, const int num_endpoints) {
        LOC_SET_t endset = LOC_SET_t();
        LOC_SET_t new_ignore_pos = LOC_SET_t();
        LOC_SET_t branch_check = LOC_SET_t();
        std::unordered_set<cmp::location>::iterator end_it, ignore_it, branch_it;
        Segment cur_seg, prog_seg;
        SEG_SET_t prog_segs = SEG_SET_t();
        SEG_SET_t::iterator seg_it;
        cmp::location loc, loc_test, loc_found;
        std::vector<Skeleton> skeleton_vec;
        int row, col, offset_row, offset_col;
        uint32_t pixel_count, pixel_ind;

        // // std::cout << "image size " << sizeof(image) << std::endl;
        // // std::cout << "should be" << rows * cols << std::endl;

        // construct set from array
        for (int ind = 0; ind < num_endpoints; ind++) {
            // reassign struct
            cmp::location endpoint;
            endpoint.row = endpoints[(ind * NUM_DIMS)];
            endpoint.col = endpoints[(ind * NUM_DIMS) + 1];
            
            // insert struct
            endset.insert(endpoint);
        }

        while (!endset.empty()) {
            // similar implementation to pop
            end_it = endset.begin();
            loc = *end_it;
            loc_test = loc;
            endset.erase(end_it);

            // create new segment
            cur_seg = Segment(loc);
            prog_segs.insert(cur_seg);

            // make a new skeleton
            Skeleton skeleton = Skeleton();
            skeleton.branch_points.clear(); // make sure to start off with empty set (paranoid :))
            skeleton.end_points.clear();
            skeleton.end_points.insert(loc); // our first endpoint is a part of this skeleton

            // iterate through current segments
            while (!prog_segs.empty()) {
                // similar implementation to pop
                seg_it = prog_segs.begin();
                prog_seg = *seg_it;
                prog_segs.erase(seg_it);

                // continue exploring this active segment
                while (1) {
                    loc = prog_seg.points.back();
                    new_ignore_pos.clear();
                    pixel_count = 0U;

                    // search nearby pixels
                    for (int pos = 0; pos < POSITIONS; pos++) {
                        // construct new pixel location
                        loc_test.row = loc.row + OFFSETS_ROW[pos];
                        loc_test.col = loc.col + OFFSETS_COL[pos];

                        // check to see if we should ignore this location or if it's valid to check
                        if (prog_seg.ignore_points.find(loc_test) == prog_seg.ignore_points.end()) {
                            // now check to see if we're within image bounds
                            if (loc_test.col >= 0 && loc_test.col < cols && loc_test.row >= 0 && loc_test.row < rows) { // we're inside the image!
                                // std::cout << "test " << loc_test.row << " " << loc_test.col << " index " << ((loc_test.row * cols) + loc_test.col) << " val " << std::flush;
                                // std::cout << ((unsigned int) (image[(loc_test.row * cols) + loc_test.col])) << std::endl;
                                if (IMAGE_LOC(image, loc_test.row, loc_test.col, cols) == HIT_MATCH) {
                                    new_ignore_pos.insert(loc_test);  // our new points to ignore on next iteration
                                    loc_found = loc_test; // for our 1-pixel match condition
                                    pixel_count++;
                                    pixel_ind = (unsigned int) pos; // apply updated index (for use in continuation)
                                }
                            }
                        }
                    }

                    // std::cout << "done " << loc.row << "," << loc.col << " count " << pixel_count << std::endl;

                    // if we think the current location is a branch (in some odd cases this can happen with right corners) let's double check it with some convolves
                    // if we hit this edge case on the edge of the image... then idk we're really SOL! In terms of bound checks let's ignore borders
                    if (pixel_count > 1 && loc.row >= 1 && loc.col >= 1 && loc.row < rows - 1 && loc.col < cols - 1) { // more than 1 bordering pixel
                        std::cout << "testing branch location  pixel count" << pixel_count << std::endl;
                        if (hitmiss::convolve_match_series(image, &BRANCH_MATCHES[0][0][0], NUM_BRANCH_MATCH, 0, 0, NUM_BRANCH_DIM, NUM_BRANCH_DIM, loc.row + BRANCH_MATRIX_OFFSET, loc.col + BRANCH_MATRIX_OFFSET, cols) == HIT_MISS) {
                            std::cout << "missed a branch! loc " << loc.row << " " << loc.col << " points " << prog_seg.num_points << std::endl;
                            pixel_count = 1U; // let's continue the path and ignore the other one (simplist way is to ignore all points except the one we're traveling in)

                            // keep track because we might have to add a lot to our ignore list
                            uint8_t found_branch = 0U;


                            // I know this isn't the most optimal solution, but because this case is so flipping rare it doesn't have to be
                            for (int pos = 0; pos < POSITIONS; pos++) {
                                // construct new pixel location
                                cmp::location t_loc;
                                t_loc.row = loc.row + OFFSETS_ROW[pos];
                                t_loc.col = loc.col + OFFSETS_COL[pos];

                                // erase from ignore as we're dealing with a new set of rules
                                new_ignore_pos.erase(t_loc);

                                // to handle yet ANOTHER edge case where the skeleton does a weird turn and then branch off check to see if we have a move that will result in a branch and make that our last point
                                if (IMAGE_LOC(image, t_loc.row, t_loc.col, cols) == HIT_MATCH && hitmiss::convolve_match_series(image, &BRANCH_MATCHES[0][0][0], NUM_BRANCH_MATCH, 0, 0, NUM_BRANCH_DIM, NUM_BRANCH_DIM, t_loc.row + BRANCH_MATRIX_OFFSET, t_loc.col + BRANCH_MATRIX_OFFSET, cols) == HIT_MATCH) {
                                    // yay! we hit a branch and can now not populate an ignore set
                                    std::cout << "ON BOARD" << t_loc.row << " " << t_loc.col << std::endl;
                                    new_ignore_pos.insert(t_loc);
                                    found_branch = 1U;
                                    loc_found = t_loc; // let's go there!
                                    pixel_ind = (unsigned int) pos; // keep track for our distance formula
                                    break;
                                }
                            }

                            // bad case of spaghetti branches and we just need to pick a random path to follow
                            if (found_branch == 0U) {
                                std::cout << "MISSING OUT" << std::endl;
                                for (int pos = 0; pos < POSITIONS; pos++) {
                                    // construct new pixel location
                                    cmp::location ig_loc;
                                    ig_loc.row = loc.row + OFFSETS_ROW[pos];
                                    ig_loc.col = loc.col + OFFSETS_COL[pos];

                                    // skip this case as we don't want to ignore it
                                    if (ig_loc == loc_found) {
                                        continue;
                                    }
                                    
                                    // ignore all other moves except this intended "non-branch-but-looks-like-branch" move
                                    new_ignore_pos.insert(ig_loc);
                                }
                            }
                        }
                    }

                    // determine if we've branched, at an end, or are continuing a path
                    if (pixel_count == 1U) { // continue path
                        // make sure we ignore the current location
                        new_ignore_pos.insert(loc);

                        // let's add the point
                        prog_seg.points.push_back(loc_found);
                        prog_seg.num_points++;
                        prog_seg.ignore_points = new_ignore_pos;
                        prog_seg.distance = prog_seg.distance + (double) DIST[pixel_ind];  // add to the segment distance
                    } else if (pixel_count == 0U) { // end node
                        // add end node to skeleton
                        skeleton.end_points.insert(loc);
                        try {
                            endset.erase(loc); // try to erase the new location
                        } catch(...) {}
                        skeleton.add_segment(prog_seg); // add location

                        // we're done with this segment
                        break;
                    } else if (skeleton.branch_points.find(loc) != skeleton.branch_points.end()) {
                        break;  // make sure we're not recursively looping through the same branch just incase it reconnects somewhere else
                    } else {
                        new_ignore_pos.insert(loc); // add current location so next branches don't detect previous branch

                        // add this segment to skeleton and remove it from our in progress list
                        skeleton.add_segment(prog_seg);

                        // add this point as a branch location to the skeleton (useful for later)
                        skeleton.branch_points.insert(loc);

                        // construct new branches
                        ignore_it = new_ignore_pos.begin();
                        while (ignore_it != new_ignore_pos.end()) {
                            cmp::location cdir = *ignore_it;
                            if (cdir != loc) { // compare locations
                                Segment new_seg = Segment(cdir); // initiate new branch
                                new_seg.ignore_points = new_ignore_pos; // make sure to ignore previous branches next time

                                // add this to our segments that are currently in progress
                                prog_segs.insert(new_seg);
                            }

                            // shift to next ignore
                            ignore_it++;
                        }

                        // we're done with the segment
                        break;
                    }
                }
            }

            // fix the skeleton by removing end points that are within the branch set
            // if iterating through a branch where two segments meet and everything hits the ignore list
            // sometimes the branch endpoint gets doubly called
            // @TODO find out why

            // add the skeleton to our vector
            skeleton_vec.push_back(skeleton);
        }

        return skeleton_vec;
    }
}