# distutils: language=c++
# cython: cdivision=True
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False


# python mods
import numpy as np
import math

# cython specific
from libcpp.vector cimport vector
from structure.analysis.hitmiss cimport convolve_match_series
from structure.analysis.treescan cimport search_skeleton, Segment, Skeleton
from structure.analysis.types cimport bool_t, uint8_t, uint32_t, int32_t, uint64_t, NPBOOL_t, NPUINT_t, NPINT32_t, NPUINT32_t, NPLONGLONG_t, NPFLOAT_t
cimport numpy as np
np.import_array()


# create python wrapper classes for the skeleton and segments of skeletons
cdef class TreeSegment(object):
    cdef Segment segment
    cdef np.ndarray points
    cdef int num_points
    cdef long double distance

    def __cinit__(self, np.int32_t[:, ::1] points, int num_points, long double distance):
        self.points = np.asarray(points)
        self.num_points = num_points
        self.distance = distance

    cdef void _set(self, Segment segment):
        self.segment = segment

    def get_points(self):
        return self.points

    def get_num_points(self):
        return self.num_points

    def __len__(self):
        return self.num_points
    
    def get_distance(self):
        return self.distance


cdef TreeSegment make_tree_segment(Segment &ref, int row_first):
    cdef int f_ind, s_ind, point_ind
    cdef np.int32_t[:, ::1] points
    cdef unsigned int num_points
    cdef long double distance
    cdef TreeSegment segment
    
    # simple copies
    num_points = ref.points.size()
    distance = ref.distance

    # reconstruct the numpy array point list (contiguous c array)
    points = np.zeros((num_points, 2), dtype=np.int32, order='c')
    with nogil:
        # specify first and second points
        point_ind = <int> 0        
        if row_first == <int> 1:
            f_ind = <int> 0
            s_ind = <int> 1
        else:
            f_ind = <int> 1
            s_ind = <int> 0

        for point in ref.points:
            points[point_ind, f_ind] = point.row
            points[point_ind, s_ind] = point.col
            point_ind += 1

    # make the python object (which has some struct structure underneath which is why need to call _set)
    segment = TreeSegment(points, num_points, distance)
    segment._set(ref)
    return segment


cdef class TreeSkeleton(object):
    cdef Skeleton *skeleton
    cdef list segments
    cdef set branches
    cdef set end_points
    cdef int row_first

    def __cinit__(self, list segments, set branches, set end_points, int row_first):
        self.segments = segments
        self.branches = branches
        self.end_points = end_points
        self.row_first = row_first

    cdef void _set(self, Skeleton *ref):
        self.skeleton = ref

    def get_segments(self):
        return self.segments

    def get_branches(self):
        return self.branches

    def get_end_points(self): 
        return self.end_points

    cpdef uint64_t get_c_obj_pointer(self) except *:
        return <uint64_t> self.skeleton

    def get_diameter(self):
        cdef Segment seg
        cdef int rf
        with nogil:
            seg = self.skeleton.get_diameter()
            rf = self.row_first
        return make_tree_segment(seg, rf)

    def __dealloc__(self):
        del self.skeleton  # let's delete the skeleton backend object


cdef TreeSkeleton make_tree_skeleton(Skeleton *ref, int row_first):
    cdef list segments
    cdef set branches
    cdef set end_points

    # reconstruct segment list for each skeleton
    segments = []
    for seg in ref.segments:
        segments.append(make_tree_segment(seg, row_first))

    # reconstruct all branches within skeleton
    branches = set()
    for branch in ref.branch_points:
        if row_first == <int> 1:
            branches.add((branch.row, branch.col))
        else:
            branches.add((branch.col, branch.row))

    # reconstruct skeleton's endpoints (different from original endpoints as they're now associated with the skeleton)
    end_points = set()
    for end in ref.end_points:
        if row_first == <int> 1:
            end_points.add((end.row, end.col))
        else:
            end_points.add((end.col, end.row))

    # make the tree skeleton and update underneath struct structure
    skeleton = TreeSkeleton(segments, branches, end_points, row_first)
    skeleton._set(ref)

    return skeleton


cpdef list search_image_skeleton(np.ndarray[NPUINT_t, ndim=2, mode='c'] image, np.ndarray[NPINT32_t, ndim=2, mode='c'] endpoints, int row_first=0):
    if image is None or endpoints is None:
        raise ValueError('Either image or endpoints are null arrays!')
    
    if image.shape[0] == 0 or endpoints.shape[0] == 0:
        raise ValueError('Cannot determine skeleton without endpoints or without image')

    # compute the c++ optimized results
    cdef vector[Skeleton*] skeletons
    cdef unsigned int[:, ::1] end_data
    end_data = np.clip(endpoints, a_min=0, a_max=None).astype(np.uint32)

    with nogil:
        skeletons = search_skeleton(&image[0, 0], &end_data[0, 0], <int> image.shape[0], <int> image.shape[1], <int> endpoints.shape[0])

    # reconstruct results into python objects
    cdef list results = []
    for skel in skeletons:
        results.append(make_tree_skeleton(skel, <int> row_first))

    # return results
    return results

'''
cpdef test_search():
    # cdef unsigned char[:, ::1] image_data
    cdef unsigned int[:, ::1] end_data
    cdef list skeleton_list = None
    cdef list segment_list = None
    cdef unsigned int point_ind
    cdef unsigned int point_size
    cdef np.ndarray point_list

    if image.shape[0] > 0:
        # image_data = image.astype(np.uint8)
        end_data = endpoints.astype(np.uint32) 
        results = search_skeleton(&image[0, 0], &end_data[0, 0], <int> image.shape[0], <int> image.shape[1], <int> end_data.shape[0])
        
        # convert results into a python object
        skeleton_list = []
        for skel in results:
            segment_list = []
            for seg in skel.segments:
                point_size = seg.points.size()
                point_list = np.zeros((point_size, 2), dtype=np.int32, order='c')
                
                # populate points 
                point_ind = <unsigned int> 0
                for point in seg.points: 
                    point_list[point_ind, 0] = point.row
                    point_list[point_ind, 1] = point.col
                    point_ind += 1

                segment_list.append({
                    'points': point_list,
                    'distance': seg.distance
                })
            skeleton_list.append({
                'segments': segment_list
            })
        
        return skeleton_list
        # return image_data
    return skeleton_list

cpdef skeleton_search(np.ndarray[NPBOOL_t, ndim=2] image, np.ndarray[NPINT32_t, ndim=2] endpoints):
    # make sure we're using a C-contiguous array for optimizations
    
    if image.flags['C_CONTIGUOUS']:
        endpoints = scan_for_end(image, rows, cols)
    else:
        endpoints = scan_for_end(np.ascontiguousarray(image), rows, cols) 


    cdef set endset = set()
    cdef Segment cur_seg, prog_seg
    cdef Segment[:] prog_segs
    cdef int[2] loc, loc_test
    cdef set ignore_pos, new_ignore_pos
    cdef list skeleton_segs
    cdef list skeleton_list
    cdef int rows, cols, row, col, offseted_row, offseted_col, pixel_count, pixel_ind

    # update size
    rows = image.shape[0]
    cols = image.shape[1]

    # create initial ignore poses
    ignore_pos = set()
    new_ignore_pos = set()
    skeleton_list = []
    # construct endset from array
    for row, col in endpoints:
        loc[0] = row
        loc[1] = col
        endset.add(loc)

    while len(endset) > 0:  # keep tracking skeletons until we run out of ends
        loc[:] = endset.pop()
        loc_test = loc

        # construct first segment
        cur_seg = make_segment(loc)  # create the first segment location
        skeleton_segs = []

        # clear in-progress segments and start with our brand new branch segment
        prog_segs = set()
        prog_segs.add(cur_seg)

        while len(prog_segs) > 0:  # keep iterating through the active segments
            prog_seg = prog_segs.pop()

            row, col = prog_seg.points[prog_seg.num_points - 1]
            loc[0] = row
            loc[1] = col
            # row = loc[0]
            # col = loc[1]
            ignore_pos = prog_seg.ignore_pos
            new_ignore_pos.clear()
            pixel_count = <int> 0

            # iterate through offset loops to determine path of skeleton
            for ind in range(POSITIONS):
                offseted_col = col + OFFSET_COLS[ind]
                offseted_row = row + OFFSET_ROWS[ind]
                loc_test[0] = offseted_row
                loc_test[1] = offseted_col

                if loc_test in ignore_pos:  # skip move if in ignore set
                    # bounds check
                    if offseted_col >= 0 and offseted_row >= 0 and offseted_col < cols and offseted_row < rows:
                        # determine if pixel exists and create segment path
                        if image[offseted_row, offseted_col] == 1:  # it exists
                            new_ignore_pos.add(loc_test)  # add to our possible ignore list
                            pixel_count += <int> 1
                            pixel_ind = <int> ind
            
            # let's just add length (all cases require this step)
            if prog_seg.num_points >= prog_seg.points.shape[0]:  # resize array if necessary
                prog_seg.points.resize((prog_seg.points.shape[0] * 2, 2))
            
            # set location
            prog_seg.points[prog_seg.num_points, :] = loc

            # increment point
            prog_seg.num_points += <int> 1
            prog_seg.distance += DIST[pixel_ind]  # fast last array value scope set

            # determine if we need to subdivide into multiple progress paths
            if pixel_count == 0:  # end node
                skeleton_segs.append(prog_seg)
                if loc in endset:
                    endset.remove(loc)  # remove that end as we've just reached it with this path
            elif pixel_count == 1:  # continuation node
                # continue this path
                prog_segs.add(prog_seg)
            else:
                # construct list of directions to ignore in next path calculation
                new_ignore_pos.add(loc)

                # ignore last branch (before overwriting var)
                skeleton_segs.append(prog_seg)

                # loop through the matched directories and create diff dirs for next path to ignore
                for cdir in new_ignore_pos:
                    if cdir != loc:
                        prog_seg = make_segment(cdir)
                        prog_seg.ignore_pos = new_ignore_pos

                        # add this branch
                        prog_segs.add(prog_seg)
        skeleton_list.append(skeleton_segs)
    return skeleton_list
'''