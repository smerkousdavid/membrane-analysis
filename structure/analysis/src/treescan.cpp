#include <cstdint>
#include <cmath>
#include <stack>
#include <set>
#include <iostream> // UNCOMMENT FOR PRINT DEBUGGING
#include "../include/treescan.hpp"
#include "../include/hitmiss.hpp"
#include "../include/hitmiss_convolves.hpp"

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
#define ORTHOG_EXTRA (long double) ORTHOG + 0.1  // slight increase (for buffer)
#define ORTHOG_MAX (long double) 2.0 * ORTHOG  // buffer for when comparing segments and how close they are (we want to be within 2 pixels of another segment)
#define MAX_DIST_BRANCH 1.5 * ORTHOG_MAX
#define MAX_FIX_PASSES 4  // max number of times to clear out single pixel segments and other common issues when scanning complex branches
#define IN_SET(SET, VAL) SET.find(VAL) != SET.end()
#define NOT_IN_SET(SET, VAL) SET.find(VAL) == SET.end()


static const double DIST[POSITIONS] = {ORTHOG, 1.0, ORTHOG, 1.0, 1.0, ORTHOG, 1.0, ORTHOG};
static const signed int OFFSETS_ROW[POSITIONS] = {-1, -1, -1, 0, 0, 1, 1, 1};
static const signed int OFFSETS_COL[POSITIONS] = {-1, 0, 1, -1, 1, -1, 0, 1};
static size_t segment_count = 0U;


namespace skeleton {
    long double loc_distance(const LOC_t p1, const LOC_t p2) {
        const long double r_diff = (long double) p1.row - (long double) p2.row;
        const long double c_diff = (long double) p1.col - (long double) p2.col;
        return std::sqrt((r_diff * r_diff) + (c_diff * c_diff));
    }

    double low_loc_distance(const LOC_t p1, const LOC_t p2) {
        const double r_diff = (double) p1.row - (double) p2.row;
        const double c_diff = (double) p1.col - (double) p2.col;
        return (double) std::sqrt((r_diff * r_diff) + (c_diff * c_diff));
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

    size_t Segment::get_id() {
        return this->id;
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

    void Segment::reverse_points() {
        this->points = this->get_points_reversed();
    }

    void Segment::extend_segment(Segment &other) {
        LOC_t loc;
        if (this->is_empty()) {
            loc.row = 0; // roll of the dice which direction we take
            loc.col = 0;
        } else {
            loc = this->get_last();
        }

        return this->extend_segment_close_to(&other, loc);
    }

    void Segment::extend_segment_close_to(Segment *other, LOC_t close) {
        if (other->is_empty()) {
            // std::cout << "empty?" << std::endl;
            return; // nothing to do
        }

        double dist_first = low_loc_distance(other->get_first(), close);
        double dist_last = low_loc_distance(other->get_last(), close);

        if (dist_first <= dist_last) {  // keep current order (our last matches their first)
            // std::cout << "foward scan" << std::endl;
            // std::cout << "forward scan " << this->points.size() << std::endl;
            for (LOC_VEC_t::iterator other_it = other->points.begin(); other_it != other->points.end(); ++other_it) {
                this->points.push_back(*other_it);
            }

            // add distance of other segment to this one including the distance between our last and their first
            this->add_distance(dist_first);
        } else {
            // add points in reverse
            // std::cout << "reverse scan " << this->points.size() << std::endl;
            for (LOC_VEC_t::reverse_iterator other_it = other->points.rbegin(); other_it != other->points.rend(); ++other_it) {
                this->points.push_back(*other_it);
            }

            // add distance of other segment to this one including the distance between our last and their "last"
            this->add_distance(dist_last);
        }

        // std::cout << "after scan " << this->points.size() << std::endl;

        // both cases (add other segment's distance and the amount of total points)
        this->add_distance(other->get_distance());
        this->num_points = this->num_points + other->num_points;
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
        } else { // more complicated as it might be the point CLOSE to it but not quite equal
            if (low_loc_distance(loc, this->get_first()) < low_loc_distance(loc, this->get_last())) {
                return this->get_last();
            }

            return this->get_first();
        }
        return loc; // there isn't an opposite of this location (not really needed)
    }

    long double Segment::get_end_distance(LOC_t loc) {
        const long double d1 = loc_distance(this->get_first(), loc);
        const long double d2 = loc_distance(this->get_last(), loc);
        return (d1 < d2) ? d1 : d2;
    }

    bool Segment::is_close_to_dist(LOC_t loc, double max_dist) {
        return this->get_end_distance(loc) <= max_dist;  // within a pixel or two
    }

    bool Segment::is_close_to(LOC_t loc) {
        return this->is_close_to_dist(loc, ORTHOG_MAX); // within a pixel or two
    }

    void Segment::set_ignore(LOC_SET_t ignore) {
        this->ignore_points = ignore;
    }

    seg_hash_triple Segment::make_hash_triple() {
        seg_hash_triple hash;
        hash.num_points = static_cast<int>(this->points.size());
        
        // handle each num points case
        switch (hash.num_points) {
            case 0:
                break;
            case 1: {
                hash.middle_middle = this->get_first();
                break;
            }
            case 2: {
                hash.middle_left = this->get_first();
                hash.middle_right = this->get_last();
                break;
            }
            default: {
                // get the middle three points
                const int ind_middle = static_cast<int>(std::floor(static_cast<long double>(hash.num_points - 1) / 2.0));
                hash.middle_left = this->get_point(ind_middle - 1);
                hash.middle_middle = this->get_point(ind_middle);
                hash.middle_right = this->get_point(ind_middle + 1);
                break;
            }
        }

        return hash;
    }

    bool Segment::is_in_seg_set(const seg_hash_triple &triple, LOC_SET_t &locs) {
        // different cases needed to be accounted for depending on the size of the diameter
        if (triple.num_points == 0) {
            return false;
        } else if (triple.num_points == 1) {
            return IN_SET(locs, triple.middle_middle); // just check middle
        } else if (triple.num_points == 2) {
            return IN_SET(locs, triple.middle_left) && IN_SET(locs, triple.middle_right); // odd case where we want both first and last points to already exist
        }

        // if any of the points exist then return true (for n >= 3)
        return ( 
            IN_SET(locs, triple.middle_left) ||
            IN_SET(locs, triple.middle_middle) || 
            IN_SET(locs, triple.middle_right)
        );
    }

    bool Segment::add_if_not_in_seg_set(LOC_SET_t &locs) {
        // add seg hash
        const seg_hash_triple triple = this->make_hash_triple();
        bool status = this->is_in_seg_set(triple, locs);

        // skip adding
        if (status) {
            return true;
        }

        // add the middle points if they don't already exist
        if (triple.num_points == 1) {
            locs.insert(triple.middle_middle);
        } else if (triple.num_points == 2) {
            locs.insert(triple.middle_left);
            locs.insert(triple.middle_right);
        } else if (triple.num_points >= 3) {
            locs.insert(triple.middle_left);
            locs.insert(triple.middle_middle);
            locs.insert(triple.middle_right);
        }

        return false;
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

    bool Segment::equals(Segment *other) {
        // return (this->points.size() == other->points.size()) && (
        //     ((this->get_first() == other->get_first()) && (this->get_last() == other->get_last())) || 
        //     ((this->get_first() == other->get_last()) && (this->get_last() == other->get_first()))
        // );
        LOC_SET_t p;
        this->add_if_not_in_seg_set(p);
        const seg_hash_triple triple = other->make_hash_triple();
        return other->is_in_seg_set(triple, p);
        // UNCOMMENT TO TEST EQUALS
        //     std::cout << "ITEMS ";
        //     for (auto it = p.cbegin(); it != p.cend(); it++) {
        //         std::cout << it->col << "," << it->row << ' ';
        //     }
        //     std::cout << " CHECKING ";
        //     std::cout << triple.middle_left.col << "," << triple.middle_left.row << " ";
        //     std::cout << triple.middle_middle.col << "," << triple.middle_middle.row << " ";
        //     std::cout << triple.middle_right.col << "," << triple.middle_right.row << " ";
        //     std::cout << std::endl;
        //     return true;
        // }

        return false;
    }

    bool Segment::not_equals(Segment *other) {
        // if (this->points.size() != other->points.size()) return false;
        // return (
        //     !((this->get_first() == other->get_first()) && (this->get_last() == other->get_last())) && 
        //     !((this->get_first() == other->get_last()) && (this->get_last() == other->get_first()))
        // );
        return !this->equals(other);
    }

    // bool Segment::contains(Segment *other) {
        
    // }

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

    bool Skeleton::is_close_to_branch(LOC_t &location, double max_dist) {
        if (this->branch_points.empty()) {
            return false; // not possible
        }

        LOC_SET_t::iterator branch_it = this->branch_points.begin();
        while (branch_it != this->branch_points.end()) {
            if (skeleton::low_loc_distance(*branch_it, location) <= max_dist) {
                return true;
            }

            branch_it++;
        }

        // didn't match any of them
        return false;
    }

    std::pair<bool, LOC_t> Skeleton::get_closest_branch(LOC_t &location) {
        if (this->branch_points.empty()) {
            LOC_t base;
            return std::pair<bool, LOC_t>(false, base); // not possible
        }

        LOC_SET_t::iterator branch_it = this->branch_points.begin();
        // sanity check (this should never happen)
        if (branch_it == this->branch_points.end()) {
            LOC_t base;
            return std::pair<bool, LOC_t>(false, base);
        }

        // first case
        std::pair<double, LOC_t> closest = std::pair<double, LOC_t>(skeleton::low_loc_distance(*branch_it, location), *branch_it);
        branch_it++;

        // continue scanning for smallest distance
        while (branch_it != this->branch_points.end()) {
            double dist = skeleton::low_loc_distance(*branch_it, location);
            if (dist <= closest.first) {
                closest = std::pair<double, LOC_t>(dist, *branch_it);
            }
            branch_it++;
        }

        return std::pair<bool, LOC_t>(true, closest.second);
    }

    bool Skeleton::follow_long_path(LOC_t start_loc, Segment *scan, Segment *full, std::unordered_set<size_t> &segs_used, LOC_SET_t &branches_used) {
        // nothing to do if empty
        if (scan->is_empty()) {
            return false;
        }
        
        // get the other side so we can find points close to this one
        LOC_t opp_loc = scan->get_opposite_loc(start_loc);
        // bool scan_close_to_branch = this->is_close_to_branch(start_loc, MAX_DIST_BRANCH);
        // std::pair<bool, LOC_t> close_branch;
        // if (scan_close_to_branch) { // if we found a close branch let's get the closest one
        //     close_branch = this->get_closest_branch(start_loc);

        //     // pre-emptive stop if the close branch has already been used
        //     if (IN_SET(branches_used, close_branch.second)) {
        //         return false; // let's stop the scan
        //     }

        //     // let's add this branch as one that's currently being used
        //     branches_used.insert(close_branch.second);
        // }

        // check the last location
        // scan_close_to_branch = this->is_close_to_branch(opp_loc, MAX_DIST_BRANCH); // REMOVE
        // if (scan_close_to_branch) { // if we found a close branch let's get the closest one
        //     close_branch = this->get_closest_branch(opp_loc);

        //     // pre-emptive stop if the close branch has already been used
        //     if (IN_SET(branches_used, close_branch.second)) {
        //         return false; // let's stop the scan
        //     }

        //     // let's add this branch as one that's currently being used
        //     branches_used.insert(close_branch.second);
        // }

        // add current scan to path
        if (full->is_empty()) {
            full->extend_segment_close_to(scan, start_loc); // extend and keep order of that close to start location
        } else {
            full->extend_segment_close_to(scan, full->get_last()); // keep the order that keeps the next segment closest to our overall path
        }
        
        // add this to our "used" endpoint list
        // ends_used.insert(scan.get_first());
        // ends_used.insert(scan.get_last());
        segs_used.insert(scan->get_id()); // we've just used the scanned segment

        // let's make sure it's not an end first
        if (IN_SET(this->end_points, opp_loc)) {
            // it's an end point so we're done with this segment
            return false; // let's not continue this path
        }

        // std::cout << "done scan end check" << std::endl;

        // let's try to find the closest segment to us by distance to first/last points
        std::vector<Segment> follow_paths;
        LOC_SET_t temp_branch_scan = branches_used;

        int duplicates = 0;
        for(std::vector<Segment>::iterator seg_it = this->segments.begin(); seg_it != this->segments.end(); ++seg_it) {
            // if (seg_it->equals(scan)) duplicates++;
            if (seg_it->is_empty() || seg_it->points.empty()) { // no point in small segments
                continue;
            // } else if (seg_it->equals(scan)) {
            //     continue; // don't scan the same segment
            } else if (IN_SET(segs_used, seg_it->get_id())) {
                continue;  // don't scan an already scanned segment
            }

            // we found a nearby segment (that hasn't been used yet)
            LOC_t first, last;
            first = seg_it->get_first();
            last = seg_it->get_last();
            // const bool not_first_used = (ends_used.find(first) == ends_used.end());
            // const bool not_last_used = (ends_used.find(last) == ends_used.end());
            
            // KEEP THIS FOR DEBUGGING BECAUSE WHO KNOWS IF WE RUN INTO THE LOOP OF DEATH AGAIN WITH SOME WEIRD BRANCHING COMBINATIONS
            // if (this->is_close_to_branch(opp_loc, 1.5) && seg_it->is_close_to(opp_loc)) {
            //     if (opp_loc == first || opp_loc == last) {
            //         std::cout << "CLOSE TO BRANCH! (" << opp_loc.col << "," << opp_loc.row << ") AND (" << first.col << "," << first.row << ") = (" << last.col << "," << last.row << ")" << std::endl;
            //     } else {
            //         std::cout << "NON-MATCH CLOSENESS" << std::endl;
            //     }
            // }

            // let's determine how close endpoints have to be if we're close to a branch or not
            double end_distance = seg_it->get_end_distance(opp_loc);
            // if (end_distance > MAX_DIST_BRANCH) {
            //     continue; // don't even consider this segment as it's too far away
            // }

            // if we're somewhat close to a branch let's consider the exclusion list 
            // const bool close_to_branch = this->is_close_to_branch(opp_loc, ORTHOG_MAX) && (this->is_close_to_branch(first, ORTHOG_MAX) || this->is_close_to_branch(last, ORTHOG_MAX));
            
            // first let's make sure that if we're close to a branching point that we check if we've used it before when scanning
            // this check will prevent us from looping over and over and scanning the same segment twice
            if (end_distance <= ORTHOG_MAX) { // && not_first_used && not_last_used) {
                // bool close_to_branch = this->is_close_to_branch(opp_loc, 1.5)
                
                Segment full_copy = full->copy();  // copy full segment to compare
                // LOC_SET_t ends_copy = ends_used;
                std::unordered_set<size_t> segs_copy = segs_used;

                // let's get which end to follow (depending on which one is closer)
                LOC_t end_follow;
                if (low_loc_distance(first, opp_loc) < low_loc_distance(last, opp_loc)) {
                    end_follow = first;
                } else {
                    end_follow = last;
                }

                this->follow_long_path(end_follow, &(*seg_it), &full_copy, segs_copy, branches_used);

                // add it to our comparator list
                follow_paths.push_back(full_copy);
            }
        }

        // std::cout << "duplicates " << duplicates << " " << this->branch_points.size() << std::endl;

        // let's find which sub-path is the longest path
        Segment *longest_seg = scan; // use current scan as the "longest" path
        for(std::vector<Segment>::iterator pair_it = follow_paths.begin(); pair_it != follow_paths.end(); ++pair_it) {
            if (pair_it->get_distance() > longest_seg->get_distance()) {
                longest_seg = &(*pair_it); // copy longest segment
                // ends_usede
            }
        }

        // copy to full report
        longest_seg->copy_to(full);

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
            std::vector<std::pair<LOC_t, Segment*>> end_follow;

            // let's construct all of the possible segment combinations starting with each segment
            LOC_SET_t::iterator end_it = this->end_points.begin();
            while (end_it != this->end_points.end()) {
                LOC_t end = *end_it;

                // scan segments and find one that matches the end
                for(std::vector<Segment>::iterator seg_it = this->segments.begin(); seg_it != this->segments.end(); ++seg_it) {
                    // found a matching segment that has this end
                    if (!seg_it->is_empty() && seg_it->is_first_or_last(end)) {
                        end_follow.push_back(std::pair<LOC_t, Segment*>(end, &(*seg_it)));
                    }
                }
                end_it++;
            }

            // now that we have the ends matched with their segments let's follow paths
            LOC_SET_t ends_used, branches_used;
            std::unordered_set<size_t> used_segments; // ids of the segments that have already been use
            Segment *longest_path;
            bool found = false;

            for(std::vector<std::pair<LOC_t, Segment*>>::iterator pair_it = end_follow.begin(); pair_it != end_follow.end(); ++pair_it) {
                // ends_used.clear(); // clear results
                used_segments.clear();
                branches_used.clear();

                Segment *follow_path = new Segment();
                found = false;
                if (this->follow_long_path(pair_it->first, pair_it->second, follow_path, used_segments, branches_used)) {  // make sure we have a successful scan
                    if (follow_path->get_distance() > diameter.get_distance()) {
                        longest_path = follow_path;  // found longest path
                        found = true;
                    }
                }

                // copy to diameter if longest was found
                if (found) {
                    this->diameter = *longest_path;
                }

                // delete follow path from heap
                delete follow_path;
            }

        } else {
            return Segment(); // return empty segment
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

                // finally let's go through all of the segments and if the segments are not close to a branch but are close to each other then connect them
                /* @TODO this implementation to connect small segments near branches is too finicky and needs to be fixed in the future
                bool rescan = true;
                while (rescan) {
                    rescan = false;
                    for(std::vector<Segment>::iterator seg_it = cur_skel->segments.begin(); (seg_it != cur_skel->segments.end() && !rescan); ++seg_it) {   
                        for(std::vector<Segment>::iterator sub_seg_it = cur_skel->segments.begin(); sub_seg_it != cur_skel->segments.end(); ++sub_seg_it) {
                            if (seg_it->get_id() == sub_seg_it->get_id() || seg_it->points.empty() || sub_seg_it->points.empty()) continue; // don't rescan same segment

                            // if their ends are close
                            LOC_t close_loc, close_loc2;
                            bool is_first = false;
                            bool reverse = false;
                            bool branch_worry = false;
                            if (skeleton::low_loc_distance(seg_it->get_first(), sub_seg_it->get_first()) <= ORTHOG_EXTRA) {
                                close_loc = sub_seg_it->get_first();
                                close_loc2 = seg_it->get_first();
                                is_first = false;
                                reverse = true;
                            } else if (skeleton::low_loc_distance(seg_it->get_first(), sub_seg_it->get_last()) <= ORTHOG_EXTRA) {
                                close_loc = seg_it->get_first();
                                close_loc2 = sub_seg_it->get_last();
                                is_first = true;
                                reverse = false;
                            } else if (skeleton::low_loc_distance(seg_it->get_last(), sub_seg_it->get_first()) <= ORTHOG_EXTRA) {
                                close_loc = sub_seg_it->get_first();
                                close_loc2 = seg_it->get_last();
                                is_first = false;
                                reverse = false;
                            } else if (skeleton::low_loc_distance(seg_it->get_last(), sub_seg_it->get_last()) <= ORTHOG_EXTRA) {
                                close_loc = seg_it->get_last();
                                close_loc2 = sub_seg_it->get_last();
                                is_first = false;
                                reverse = false; 
                            } else {
                                continue; // segments do not align
                            }

                            // if the segments line up let's check the branches
                            std::pair<bool, LOC_t> branch = cur_skel->get_closest_branch(close_loc);
                            if (branch.first) { // branch found
                                double dist1 = skeleton::low_loc_distance(branch.second, close_loc);
                                double dist2 = skeleton::low_loc_distance(branch.second, close_loc2);

                                // if distance to closest branch is met by the two points then let's ignore it
                                if (dist1 <= ORTHOG_MAX && dist2 <= ORTHOG_MAX) {
                                    continue;
                                }
                            }

                            // finally extend and remove the corresponding segments
                            if (is_first) {
                                if (reverse) sub_seg_it->reverse_points();
                                sub_seg_it->extend_segment_close_to(&(*seg_it), close_loc); // extend 
                                seg_it = cur_skel->segments.erase(seg_it); // erase and shift pointer
                                rescan = true;
                                break; // rescan
                            } else {
                                if (reverse) seg_it->reverse_points();
                                seg_it->extend_segment_close_to(&(*sub_seg_it), close_loc);
                                sub_seg_it = cur_skel->segments.erase(sub_seg_it); // erase and shift pointer
                                rescan = true;
                                break; // rescan
                            }
                        }
                    }
                }*/

                // increment rep index
                index++;
            }

            // we want to quit the main check loop eventually
            passes++;
        }
    }

    // void copy_branch_data_to_skeleton(std::unordered_set<branch_data> &data, Skeleton *skel) {
    //     std::unordered_set<branch_data>::iterator b_it = data.begin();
    //     while (b_it != data.end()) {
    //         skel->branch_points.insert(b_it->loc);
    //         std::cout << "ADDING" << b_it->loc.col << " " << b_it->loc.row << " : " << static_cast<int>(b_it->expect_in) << std::endl;
    //         b_it++;
    //     }
    // }

    // std::pair<double, branch_data> get_closest_branch_data(std::unordered_set<branch_data> &data, LOC_t &loc) {
    //     std::unordered_set<branch_data>::iterator b_it = data.begin();

    //     // make sure we have data
    //     if (b_it == data.end()) {
    //         branch_data empty_data;
    //         return std::pair<double, branch_data>(-1.0, empty_data);
    //     }

    //     // get first distance
    //     branch_data closest = *b_it;
    //     double dist = low_loc_distance(closest.loc, loc);
    //     b_it++; // shift to the next location

    //     // keep scanning to get the closest one
    //     while (b_it != data.end()) {
    //         double ndist = low_loc_distance(b_it->loc, loc);
    //         if (ndist < dist) {
    //             dist = ndist;
    //             closest = *b_it;
    //         }
    //         b_it++;
    //     }

    //     return std::pair<double, branch_data>(dist, closest);
    // }

    // bool subtract_branch_expected(std::unordered_set<branch_data> &data, LOC_t &loc, bool do_subtract=true) {
    //     // true if needed to break (all expected gone)
    //     std::pair<double, branch_data> close = get_closest_branch_data(data, loc);
    //     if (close.first >= 0 && close.first <= ORTHOG_EXTRA) { // some expected branch inputs
    //         if (close.second.expect_in == 0 || close.second.expect_in > 10) { // we ran out! (possibly overflowed)
    //             return true; // let's quit whatever outer loop we're running
    //         }

    //         if (do_subtract) {
    //             // get current data and update the list
    //             branch_data bdata = close.second;

    //             // remove if found current data
    //             if (data.find(bdata) != data.end()) {
    //                 data.erase(bdata);
    //             }

    //             // update the data
    //             bdata.expect_in = bdata.expect_in - 1; // subtract one for the expected amount of inputs
    //             data.insert(bdata); // hashed by location so let's update the data
    //         }
    //     }

    //     return false;
    // }

    // bool subtract_branch_pair(std::unordered_set<branch_data> &data, LOC_t &loc_1, LOC_t &loc_2, bool do_subtract=true) {
    //     if (do_subtract) {
    //         if (subtract_branch_expected(data, loc_1, false) || subtract_branch_expected(data, loc_2, false)) {
    //             return true; // we failed initial test
    //         }

    //         // do the actual subtract and return result of second test
    //         subtract_branch_expected(data, loc_1, true);
    //         subtract_branch_expected(data, loc_2, true);
    //         return false;
    //     }

    //     // we're just doing a simple test
    //     return subtract_branch_expected(data, loc_1, false) || subtract_branch_expected(data, loc_2, false);
    // }

    std::vector<Skeleton*> search_skeleton(const uint8_t* image, const uint32_t* endpoints, const int rows, const int cols, const int num_endpoints) {
        LOC_SET_t endset = LOC_SET_t();
        LOC_SET_t new_ignore_pos = LOC_SET_t();
        LOC_SET_t branch_check = LOC_SET_t();
        // std::unordered_set<branch_data> skel_branches;
        LOC_SET_t seg_hashes; // segment hashes to make sure we don't rescan segments that are in loops
        LOC_SET_t critical_ignore; // critical branching points that should be ignored when adding newly scanned branches
        std::unordered_set<cmp::location>::iterator end_it, ignore_it, branch_it;
        Segment cur_seg, prog_seg;
        SEG_SET_t prog_segs = SEG_SET_t();
        SEG_SET_t::iterator seg_it;
        cmp::location loc, loc_test, loc_found;
        std::vector<Skeleton*> skeleton_vec;
        int row, col, offset_row, offset_col;
        double pixel_dist;
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
            
            // temporary holder for branch points
            // skel_branches.clear();

            // iterate through current segments
            while (!prog_segs.empty()) {
                // similar implementation to pop
                seg_it = prog_segs.begin();
                prog_seg = *seg_it;
                prog_segs.erase(seg_it);

                // let's first check to make sure this segment needs to be scanned
                // if (!prog_seg.is_empty() && subtract_branch_pair(skel_branches, prog_seg.get_first(), prog_seg.get_last(), false)) {
                //     break; // we're not supposed to scan this!
                // }

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
                        if (NOT_IN_SET(prog_seg.ignore_points, loc_test)) {
                            // now check to see if we're within image bounds
                            if (loc_test.col >= 0 && loc_test.col < cols && loc_test.row >= 0 && loc_test.row < rows) { // we're inside the image!
                                // std::cout << "test " << loc_test.row << " " << loc_test.col << " index " << ((loc_test.row * cols) + loc_test.col) << " val " << std::flush;
                                // std::cout << ((unsigned int) (image[(loc_test.row * cols) + loc_test.col])) << std::endl;
                                if (IMAGE_LOC(image, loc_test.row, loc_test.col, cols) == HIT_MATCH) {
                                    new_ignore_pos.insert(loc_test);  // our new points to ignore on next iteration
                                    loc_found = loc_test; // for our 1-pixel match condition
                                    pixel_count++;
                                    pixel_ind = (unsigned int) pos; // apply updated index (for use in continuation)
                                    // pixel_dist = DIST[pos];
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
                    if (pixel_count > 1U) { // branching location
                        /*
                        // get critical points that were scanned (to check if it's a weird branch)
                        LOC_SET_t intersect;
                        ignore_it = new_ignore_pos.begin();
                        while (ignore_it != new_ignore_pos.end()) { // find the set intersection of the two unordered sets O(n)
                            if (IN_SET(critical_ignore, *ignore_it)) {
                                intersect.insert(*ignore_it);
                            }
                            ignore_it++;
                        }
                        size_t critical = intersect.size();

                        // special case where if there is only 1 or 0 paths left (a continuation) then don't branch out
                        int diff = (static_cast<int>(new_ignore_pos.size()) - static_cast<int>(critical));
                        if (diff <= 1) {
                            pixel_count = 0U; // end node
                        } else if (diff == 2) {
                            pixel_count = 1U; // continue path

                            // weird case where this path has already scanned critical points so find the new continuation path
                            ignore_it = new_ignore_pos.begin();
                            while (ignore_it != new_ignore_pos.end()) {
                                cmp::location cdir = *ignore_it;
                                if (NOT_IN_SET(critical_ignore, cdir)) {
                                    loc_found = cdir;
                                    pixel_dist = skeleton::low_loc_distance(loc, loc_found); // approximate new pixel distance
                                    break; // found the first non-critical continuation path
                                }
                                ignore_it++;
                            }
                        } else {*/
                            // check that the segment hasn't already been scanned
                            if (prog_seg.add_if_not_in_seg_set(seg_hashes)) {
                                break;
                            }
                            // add this segment to skeleton and remove it from our in progress list
                            skeleton->add_segment(prog_seg);

                            // construct new branches
                            ignore_it = new_ignore_pos.begin();
                            uint8_t count = 0;
                            while (ignore_it != new_ignore_pos.end()) {
                                cmp::location cdir = *ignore_it;
                                if (cdir != loc && NOT_IN_SET(critical_ignore, cdir)) { // compare locations
                                    Segment new_seg = Segment(cdir); // initiate new branch
                                    new_seg.ignore_points = new_ignore_pos; // make sure to ignore previous branches next time
                                    new_seg.ignore_points.insert(loc); // add current point as an ignore

                                    // add this to our segments that are currently in progress
                                    prog_segs.insert(new_seg);

                                    // add this as a critical point
                                    critical_ignore.insert(cdir);

                                    // to add to expected input
                                    count++;
                                }

                                // shift to next ignore
                                ignore_it++;
                            }

                            // add this point as a branch location to the skeleton (useful for later)
                            if (!skeleton->is_close_to_branch(loc, ORTHOG_EXTRA)) {
                                skeleton->branch_points.insert(loc);
                            }

                            // we're done with the segment
                            break;
                        // }
                    }

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

                        // make sure this segment has not been scanned yet
                        if (prog_seg.add_if_not_in_seg_set(seg_hashes)) {
                            break;
                        }

                        try {
                            endset.erase(loc); // try to erase the new location
                        } catch(...) {}
                        skeleton->add_segment(prog_seg); // add location
    
                        // we're done with this segment
                        break;
                    } else {
                        std::cout << "WARNING! Reached a multi-pixel point which should not be possible please check treescan.cpp" << std::endl;
                    }
                }
            }

            // fix the skeleton by removing end points that are within the branch set
            // if iterating through a branch where two segments meet and everything hits the ignore list
            // sometimes the branch endpoint gets doubly called
            // @TODO find out why

            // copy the branch locations to the skeleton
            // copy_branch_data_to_skeleton(skel_branches, skeleton);

            // add the skeleton to our vector
            skeleton->diameter_dirty = true; // new points! so it's definetely dirty
            skeleton_vec.push_back(skeleton);
        }

        // fix results before returning (removing single pixel segments)
        // fix_skeleton(&skeleton_vec);

        return skeleton_vec;
    }
}