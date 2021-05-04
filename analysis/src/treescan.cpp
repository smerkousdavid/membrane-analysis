#include <cstdint>
#include <cmath>
#include <stack>
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
#define ORTHOG (long double) sqrt(2)
#define ORTHOG_MAX (long double) 2.0 * ORTHOG  // buffer for when comparing segments and how close they are (we want to be within 2 pixels of another segment)
#define MAX_FIX_PASSES 4  // max number of times to clear out single pixel segments and other common issues when scanning complex branches
#define NUM_BRANCH_MATCH 19 // number of convolve branch matches
#define NUM_BRANCH_DIM 3 // dimensions of the branch hits

static const long double DIST[POSITIONS] = {ORTHOG, 1.0, ORTHOG, 1.0, 1.0, ORTHOG, 1.0, ORTHOG};
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
    long double loc_distance(const LOC_t p1, const LOC_t p2) {
        const long double r_diff = (long double) p1.row - (long double) p2.row;
        const long double c_diff = (long double) p1.col - (long double) p2.col;
        return std::sqrt((r_diff * r_diff) + (c_diff * c_diff));
    }
    
    /* SEGMENT START */
    Segment::Segment() {
        this->construct();
    }

    Segment::Segment(LOC_t initial_loc) {
        this->construct();
        this->add_point(initial_loc);
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

    void Segment::add_point(LOC_t point) {
        this->points.push_back(point);
        this->num_points++;
    }

    void Segment::add_distance(long double dist) {
        this->distance = this->distance + dist;
    }

    bool Segment::is_empty() {
        return this->points.empty();
    }

    LOC_t Segment::get_point(size_t pos) {
        return this->points.at(pos);
    }

    LOC_t Segment::get_first() {
        return this->get_point(0);
    }

    long double Segment::get_distance() {
        return this->distance;
    }

    LOC_t Segment::get_last() {
        return this->get_point(this->points.size() - 1);
    }

    LOC_VEC_t Segment::get_points() {
        return this->points;
    }

    LOC_VEC_t Segment::get_points_reversed() {
        LOC_VEC_t rev_points;

        // return empty vector
        if (this->is_empty()) {
            return rev_points;
        }

        LOC_VEC_t::reverse_iterator loc_it = this->points.rbegin();
        while(loc_it != this->points.rend()) {
            rev_points.push_back((LOC_t) *loc_it);
            loc_it++;
        }

        return rev_points;
    }

    void Segment::extend_segment(Segment &other) {
        LOC_t loc;
        if (this->is_empty()) {
            loc.row = 0; // roll of the dice which direction we take
            loc.col = 0;
        } else {
            loc = this->get_last();
        }

        return this->extend_segment_close_to(other, loc);
    }

    void Segment::extend_segment_close_to(Segment &other, LOC_t close) {
        if (other.is_empty()) {
            // std::cout << "empty?" << std::endl;
            return; // nothing to do
        }

        long double dist_first = loc_distance(other.get_first(), close);
        long double dist_last = loc_distance(other.get_last(), close);

        if (dist_first <= dist_last) {  // keep current order (our last matches their first)
            // std::cout << "foward scan" << std::endl;
            // std::cout << "forward scan " << this->points.size() << std::endl;
            for (LOC_VEC_t::iterator other_it = other.points.begin(); other_it != other.points.end(); ++other_it) {
                this->points.push_back(*other_it);
            }

            // add distance of other segment to this one including the distance between our last and their first
            this->add_distance(dist_first);
        } else {
            // add points in reverse
            // std::cout << "reverse scan " << this->points.size() << std::endl;
            for (LOC_VEC_t::reverse_iterator other_it = other.points.rbegin(); other_it != other.points.rend(); ++other_it) {
                this->points.push_back(*other_it);
            }

            // add distance of other segment to this one including the distance between our last and their "last"
            this->add_distance(dist_last);
        }

        // std::cout << "after scan " << this->points.size() << std::endl;

        // both cases (add other segment's distance and the amount of total points)
        this->add_distance(other.get_distance());
        this->num_points = this->num_points + other.num_points;
    }

    bool Segment::is_first_or_last(LOC_t point) {
        if (this->is_empty()) {
            return 0;  // can't be because we have no points
        }

        // check to see if this point is a part of the "ends" of the segment
        return (this->get_first() == point) || (this->get_last() == point); 
    }

    LOC_t Segment::get_opposite_loc(LOC_t loc) {
        if (this->is_empty()) {
            return loc; // nothing to do
        }

        if (this->get_first() == loc) {
            return this->get_last(); // opposite of first location
        } else if (this->get_last() == loc) {
            return this->get_first(); // opposite of last location
        }
        return loc; // there isn't an opposite of this location
    }

    long double Segment::get_end_distance(LOC_t loc) {
        const long double d1 = loc_distance(this->get_first(), loc);
        const long double d2 = loc_distance(this->get_last(), loc);
        return (d1 < d2) ? d1 : d2;
    }

    bool Segment::is_close_to(LOC_t loc) {
        return this->get_end_distance(loc) <= ORTHOG_MAX;  // within a pixel or two
    }

    void Segment::set_ignore(LOC_SET_t ignore) {
        this->ignore_points = ignore;
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

    void Segment::copy_to(Segment *other) {
        other->id = this->id;
        other->num_points = this->num_points;
        other->ignore_points = this->ignore_points;
        other->distance = this->distance;
        other->points = this->points;
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
        this->diameter_dirty = true;
    }

    bool Skeleton::is_empty() {
        return this->segments.empty() || this->end_points.empty();
    }

    bool Skeleton::follow_long_path(LOC_t start_loc, Segment &scan, Segment *full, LOC_SET_t &ends_used) {
        // nothing to do if empty
        if (scan.is_empty()) {
            return false;
        }
        
        // get the other side so we can find points close to this one
        LOC_t opp_loc = scan.get_opposite_loc(start_loc);

        // std::cout << "attempt extend" << std::endl;

        // add current scan to path
        if (full->is_empty()) {
            full->extend_segment_close_to(scan, start_loc); // extend and keep order of that close to start location
        } else {
            full->extend_segment_close_to(scan, full->get_last()); // keep the order that keeps the next segment closest to our overall path
        }
        
        // add this to our "used" endpoint list
        ends_used.insert(scan.get_first());
        ends_used.insert(scan.get_last());

        // let's make sure it's not an end first
        if (this->end_points.find(opp_loc) != this->end_points.end()) {
            // it's an end point so we're done with this segment
            return false; // let's not continue this path
        }

        // std::cout << "done scan end check" << std::endl;

        // let's try to find the closest segment to us by distance to first/last points
        std::vector<std::pair<Segment, LOC_SET_t>> follow_paths;
        for(std::vector<Segment>::iterator seg_it = this->segments.begin(); seg_it != this->segments.end(); ++seg_it) {
            if (seg_it->is_empty() || seg_it->get_distance() == 0.0) {
                continue;
            }

            // we found a nearby segment (that hasn't been used yet)
            LOC_t first, last;
            first = seg_it->get_first();
            last = seg_it->get_last();
            if (seg_it->is_close_to(opp_loc) && (ends_used.find(first) == ends_used.end()) && (ends_used.find(last) == ends_used.end())) {
                Segment full_copy = full->copy();  // copy full segment to compare
                LOC_SET_t ends_copy = ends_used;
                
                // let's get which end to follow (depending on which one is closer)
                LOC_t end_follow;
                if (loc_distance(first, opp_loc) < loc_distance(last, opp_loc)) {
                    end_follow = first;
                } else {
                    end_follow = last;
                }

                // std::cout << "    FOLLOW " << end_follow.row << " " << end_follow.col << " dist " << seg_it->get_end_distance(opp_loc) << " OPPOSITE " << opp_loc.row << " " << opp_loc.col << std::endl;
                
                // follow the path of the current segment and add it to our full "copy"
                // std::cout << "path before had " << full_copy.points.size() << " points! however full has" << full->points.size()  << std::endl;
                ends_copy.insert(full_copy.get_first());
                ends_copy.insert(full_copy.get_last());
                ends_copy.insert(end_follow);
                this->follow_long_path(end_follow, *seg_it, &full_copy, ends_copy);

                // std::cout << "path now has " << full_copy.points.size() << " points!" << std::endl;

                // add it to our comparator list
                follow_paths.push_back(std::pair<Segment, LOC_SET_t>(full_copy, ends_copy));
            }
        }

        // let's find which sub-path is the longest path
        Segment longest_seg = scan; // use current scan as the "longest" path
        for(std::vector<std::pair<Segment, LOC_SET_t>>::iterator pair_it = follow_paths.begin(); pair_it != follow_paths.end(); ++pair_it) {
            if (pair_it->first.get_distance() > longest_seg.get_distance()) {
                longest_seg = pair_it->first; // copy longest segment
                // ends_used
            }
        }

        // copy to full report
        longest_seg.copy_to(full);

        return true; // we've finished the path (used just in case our seg scans of the skeletons are all empty)
    }

    Segment Skeleton::get_diameter() {
        // if we're not dirty let's return the previously calculated diameter
        if (!this->diameter_dirty) {
            return this->diameter;
        }

        // Segment diameter;

        // let's only work with a valid segment list
        if (!this->is_empty()) {
            // let's handle a special case of no branching to skip all of the branch following nonsense
            if (this->segments.size() == 1) {
                return this->segments.at(0);  // get the first segment and we're done :)
            }

            // since each endpoint will follow a path we need segments for each path to keep track of points and distance
            std::vector<std::pair<LOC_t, Segment>> end_follow;

            // let's construct all of the possible segment combinations starting with each segment
            LOC_SET_t::iterator end_it = this->end_points.begin();
            while (end_it != this->end_points.end()) {
                LOC_t end = *end_it;

                // scan segments and find one that matches the end
                for(std::vector<Segment>::iterator seg_it = this->segments.begin(); seg_it != this->segments.end(); ++seg_it) {
                    // found a matching segment that has this end
                    if (!seg_it->is_empty() && seg_it->is_first_or_last(end)) {
                        end_follow.push_back(std::pair<LOC_t, Segment>(end, *seg_it));
                    }
                }
                end_it++;
            }

            // now that we have the ends matched with their segments let's follow paths
            LOC_SET_t ends_used;
            for(std::vector<std::pair<LOC_t, Segment>>::iterator pair_it = end_follow.begin(); pair_it != end_follow.end(); ++pair_it) {
                ends_used.clear(); // clear results
                
                // std::cout << "PAIR!" << std::endl;

                Segment follow_path;
                if (this->follow_long_path(pair_it->first, pair_it->second, &follow_path, ends_used)) {  // make sure we have a successful scan
                    if (follow_path.get_distance() > diameter.get_distance()) {
                        this->diameter = follow_path;  // found longest path
                    }
                }
            }
        }

        // make sure it's no longer dirty so use the cached version if possible
        this->diameter_dirty = false;

        return this->diameter;
    }
    /* SKELETON END */

    void fix_skeleton(std::vector<Skeleton*> *skeleton) {
        /** handles fixing odd things about the skeleton, such as segments that are tiny */

        // find all empty segments
        int passes = 0; 
        bool any_empty = true;
        while (any_empty && passes <= MAX_FIX_PASSES) { // max of 3 checks
            any_empty = false; // keep going until there are no more empty segments
            // std::cout << "PASS" << std::endl;
            
            int index = 0;
            for(std::vector<Skeleton*>::iterator skel_it = skeleton->begin(); skel_it != skeleton->end(); ++skel_it) {
                Skeleton *cur_skel = *skel_it;
                
                // skip if there aren't any segments
                if (cur_skel->is_empty()) {
                    continue;
                }

                // populate a list of empty segments
                std::vector<LOC_t> empty_segs;

                // iterate through all segments
                std::vector<Segment>::iterator seg_it = cur_skel->segments.begin();
                while(seg_it != cur_skel->segments.end()) {
                    if (seg_it->points.size() == 1) { // assumed to have only one point (so no theoritical distance)
                        empty_segs.push_back(seg_it->get_first()); // deref and copy to empty segs
                        cur_skel->diameter_dirty = true;
                        seg_it = cur_skel->segments.erase(seg_it); // erase and return new valid pointer
                    } else {
                        seg_it++;
                    }
                }

                // more empty ones :(
                if (!empty_segs.empty()) {
                    any_empty = true;
                }

                // shift through all of the empty segs and continue to find replacements
                bool dirty = false; // why copy ref when we don't need to
                std::vector<LOC_t>::iterator empt_it = empty_segs.begin();
                // std::cout << "EMPTY" << std::endl;
                while (empt_it != empty_segs.end()) {
                    // Segment cur_seg = *empt_it;
                    // get the one and only point
                    // std::cout << "SCANNER" << std::endl;
                    LOC_t fix_loc = *empt_it;

                    // find a close segment (let's try being really strict at first and then ease up if there are no other matches)
                    int add_ind = -1;
                    bool add_front = false;
                    bool found = false;
                    long double found_distance = 0.0;
                    for (long double match_dist = 1.0; match_dist <= 3.0; match_dist++) {
                        int cur_ind = 0;
                        for(std::vector<Segment>::iterator seg_it = cur_skel->segments.begin(); seg_it != cur_skel->segments.end(); ++seg_it) {   
                            long double measure_distance = loc_distance(fix_loc, seg_it->get_first());
                            if (measure_distance <= match_dist) { // within 1-2 pixels
                                add_ind = cur_ind; 
                                found = true;
                                add_front = true;  // don't reverse the segment when adding it to the global list
                                found_distance = measure_distance; // use this distance to add to the total segment length
                                break;
                            } else {
                                measure_distance = loc_distance(fix_loc, seg_it->get_last());
                                if(measure_distance <= match_dist) {
                                    add_ind = cur_ind; 
                                    found = true;
                                    add_front = false;
                                    found_distance = measure_distance; // use this distance to add to the total segment length
                                    break;
                                }
                            } // else is that this segment didn't match any of the distance conditions above

                            cur_ind++;
                        }

                        // exit fix loop
                        if (found) break;
                    }

                    // std::cout << "SCANNED" << std::endl;

                    // if we found a matching segment let's update it and replace it in our skeleton
                    if (found && add_ind >= 0) {
                        Segment &ref_seg = cur_skel->segments.at(add_ind);
                        cur_skel->diameter_dirty = true;
                        if (add_front) {
                            ref_seg.points.insert(ref_seg.points.begin(), fix_loc); // add to the beginning
                        } else {
                            ref_seg.points.push_back(fix_loc);
                        }

                        // let's add this distance between the two points
                        ref_seg.add_distance(found_distance);

                        // we're dirty :)
                        dirty = true;
                    }

                    // remove elem
                    empt_it = empty_segs.erase(empt_it);
                }

                // increment rep index
                index++;
            }

            // we want to quit the main check loop eventually
            passes++;
        }
    }

    std::vector<Skeleton*> search_skeleton(const uint8_t* image, const uint32_t* endpoints, const int rows, const int cols, const int num_endpoints) {
        LOC_SET_t endset = LOC_SET_t();
        LOC_SET_t new_ignore_pos = LOC_SET_t();
        LOC_SET_t branch_check = LOC_SET_t();
        std::unordered_set<cmp::location>::iterator end_it, ignore_it, branch_it;
        Segment cur_seg, prog_seg;
        SEG_SET_t prog_segs = SEG_SET_t();
        SEG_SET_t::iterator seg_it;
        cmp::location loc, loc_test, loc_found;
        std::vector<Skeleton*> skeleton_vec;
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
            Skeleton *skeleton = new Skeleton();
            skeleton->branch_points.clear(); // make sure to start off with empty set (paranoid :))
            skeleton->end_points.clear();
            skeleton->end_points.insert(loc); // our first endpoint is a part of this skeleton

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
                    /*if (pixel_count > 1 && loc.row >= 1 && loc.col >= 1 && loc.row < rows - 1 && loc.col < cols - 1) { // more than 1 bordering pixel
                        std::cout << "testing branch location " << loc.row << " " << loc.col << " pixel count" << pixel_count << std::endl;
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
                                    new_ignore_pos.insert(loc); // ignore current
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
                    }*/

                    // determine if we've branched, at an end, or are continuing a path
                    if (pixel_count == 1U) { // continue path
                        // make sure we ignore the current location
                        new_ignore_pos.insert(loc);

                        // let's add the point
                        prog_seg.add_point(loc_found);
                        prog_seg.add_distance((double) DIST[pixel_ind]);  // add to the segment distance
                        prog_seg.set_ignore(new_ignore_pos);
                    } else if (pixel_count == 0U) { // end node
                        // add end node to skeleton
                        skeleton->end_points.insert(loc);
                        try {
                            endset.erase(loc); // try to erase the new location
                        } catch(...) {}
                        skeleton->add_segment(prog_seg); // add location

                        // we're done with this segment
                        break;
                    } else if (skeleton->branch_points.find(loc) != skeleton->branch_points.end()) {
                        break;  // make sure we're not recursively looping through the same branch just incase it reconnects somewhere else
                    } else {
                        new_ignore_pos.insert(loc); // add current location so next branches don't detect previous branch

                        // add this segment to skeleton and remove it from our in progress list
                        skeleton->add_segment(prog_seg);

                        // add this point as a branch location to the skeleton (useful for later)
                        skeleton->branch_points.insert(loc);

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
            skeleton->diameter_dirty = true; // new points! so it's definetely dirty
            skeleton_vec.push_back(skeleton);
        }

        // fix results before returning (removing single pixel segments)
        fix_skeleton(&skeleton_vec);

        return skeleton_vec;
    }
}