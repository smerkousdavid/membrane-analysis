# distutils: language=c++
# cython: cdivision=True
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from structure.analysis.hitmiss cimport convolve_match_image, location
from structure.analysis.types cimport uint8_t, uint32_t, int32_t, NPBOOL_t, NPUINT_t, NPINT32_t, NPUINT32_t, NPLONGLONG_t, NPFLOAT_t
np.import_array()


# define simple coordinate structure for images
cdef struct s_Coord:
    Py_ssize_t x
    Py_ssize_t y


# define struct
ctypedef s_Coord Coord
cdef size_t COORD_SIZE = sizeof(Coord)
cdef NPBOOL_t FIND_VAL = <NPBOOL_t> 1
cdef int NOT_FOUND = <int> 0
cdef int FOUND = <int> 1
cdef np.ndarray END_POINTS = np.array(
    [
        [[0, 0, 0],
        [0, 1, 0],
        [2, 1, 2]],

        [[0, 0, 0],
        [0, 1, 2],
        [0, 2, 1]],

        [[0, 0, 2],
        [0, 1, 1],
        [0, 0, 2]],

        [[0, 2, 1],
        [0, 1, 2],
        [0, 0, 0]],

        [[2, 1, 2],
        [0, 1, 0],
        [0, 0, 0]],

        [[1, 2, 0],
        [2, 1, 0],
        [0, 0, 0]],

        [[2, 0, 0],
        [1, 1, 0],
        [2, 0, 0]],

        [[0, 0, 0],
        [2, 1, 0],
        [1, 2, 0]]
    ],
    dtype=np.uint8,
    order='C'  # C (row) memory layout
)
cdef np.ndarray BRANCH_POINTS = np.array(
    [
        [[0, 1, 0],
        [1, 1, 1],  # X
        [0, 1, 0]],

        [[1, 0, 1],
        [0, 1, 0],  # Rotated X
        [1, 0, 1]],

        [[2, 1, 2],
        [1, 1, 1],  # T
        [2, 2, 2]],

        [[1, 2, 1],
        [2, 1, 2],
        [1, 2, 2]],

        [[2, 1, 2],
        [1, 1, 2],
        [2, 1, 2]],

        [[1, 2, 2],
        [2, 1, 2],
        [1, 2, 1]],

        [[2, 2, 2],
        [1, 1, 1],
        [2, 1, 2]],

        [[2, 2, 1],
        [2, 1, 2],
        [1, 2, 1]],

        [[2, 1, 2],
        [2, 1, 1],
        [2, 1, 2]],

        [[1, 2, 1],
        [2, 1, 2],
        [2, 2, 1]],

        [[1, 0, 1],
        [0, 1, 0],  # Y
        [2, 1, 2]],

        [[0, 1, 0],
        [1, 1, 2],
        [0, 2, 1]],

        [[1, 0, 2],
        [0, 1, 1],
        [1, 0, 2]],

        [[1, 0, 2],
        [0, 1, 1],
        [1, 0, 2]],

        [[0, 2, 1],
        [1, 1, 2],
        [0, 1, 0]],

        [[2, 1, 2],
        [0, 1, 0],
        [1, 0, 1]],

        [[1, 2, 0],
        [2, 1, 1],
        [0, 1, 0]],

        [[2, 0, 1],
        [1, 1, 0],
        [2, 0, 1]],

        [[0, 1, 0],
        [2, 1, 1],
        [1, 2, 0]]
    ],
    dtype=np.uint8,
    order='C'  # C (row) memory layout
)
cdef Py_ssize_t NUM_END_POINTS = END_POINTS.shape[0]
cdef Py_ssize_t NUM_BRANCH_POINTS = BRANCH_POINTS.shape[0]

''' Old implementation for reference. Please look at hitmiss.cpp in the src folder
cdef int scan_for_px(const NPBOOL_t[:, ::1] image, const Py_ssize_t rows, const Py_ssize_t cols, Coord* start_loc, Coord* loc, NPBOOL_t search):
    cdef Py_ssize_t row
    cdef Py_ssize_t col

    # keep going until value passes threshold
    for row in range(start_loc.y, rows):
        for col in range(start_loc.x, cols):
            if image[row, col] == search:
                loc.x = col
                loc.y = row
                return FOUND
    
    # didn't find anything :(
    loc.x = 0
    loc.y = 0
    return NOT_FOUND


# counts the number of ones in a matrix (to determine convolve match)
cdef int count_ones(NPUINT_t[:, ::1] mat,  const Py_ssize_t rows, const Py_ssize_t cols):
    cdef int count = 0
    cdef Py_ssize_t r, c
    for r in range(rows):
        for c in range(cols):
            if mat[r, c] == <unsigned int> 1:
                count += 1
    return count


# applies convolution to match matrix
cpdef int convolve_match(const NPBOOL_t[:, ::1] mat, const NPUINT_t[:, ::1] match, const Py_ssize_t start_row, const Py_ssize_t start_col, const Py_ssize_t rows, const Py_ssize_t cols):
    cdef Py_ssize_t r, c
    for r in range(rows):
        for c in range(cols):
            if match[r + start_row, c + start_col] != <unsigned int> 2 and mat[r, c] != <NPBOOL_t> match[r + start_row, c + start_col]:
                return NOT_FOUND
    return FOUND


# applies convolve match to multiple examples (until it finds the first match)
cpdef int convolve_match_series(const NPBOOL_t[:, ::1] mat, const NPUINT_t[:, :, ::1] matches, const Py_ssize_t match_num, const Py_ssize_t start_row, const Py_ssize_t start_col, const Py_ssize_t rows, const Py_ssize_t cols):
    cdef Py_ssize_t mn
    
    # test each matrix "convolution" (not exactly) until we match one
    for mn in range(match_num):
        if convolve_match(mat, matches[mn, :, :], start_row, start_col, rows, cols) == <int> 1:
            return FOUND
    
    # non of them matched
    return NOT_FOUND


cdef list scan_for_edge_end(const NPBOOL_t[:, ::1] image, const Py_ssize_t rows, const Py_ssize_t cols, Py_ssize_t start, Py_ssize_t end, const int left, const int right, const int top, const int bottom):
    cdef Py_ssize_t it
    cdef Py_ssize_t last_row, last_col, second_last_row, second_last_col
    last_row, last_col, second_last_row, second_last_col = rows - 1, cols - 1, rows - 2, cols - 2
    cdef list ends_found = []
    cdef Coord found

    if left == 1:
        if start == 0:  # top left 2x2
            if image[0, 0] == FIND_VAL and (<int> image[1, 0] + <int> image[1, 1] + <int> image[0, 1]) == FOUND:
                found.x = 0
                found.y = 0
                ends_found.append(found)
                # return FOUND 

        if end == last_row:  # bottom left 2x2
            if image[last_row, 0] == FIND_VAL and (<int> image[second_last_row, 0] + <int> image[second_last_row, 1] + <int> image[last_row, 1]) == FOUND:
                found.x = 0
                found.y = last_row
                ends_found.append(found)
                # return FOUND

        # continue to iterate by 2x3
        if end - 1 >= start + 1:
            for it in range(start + 1, end - 1):
                if image[it, 0] == FIND_VAL and convolve_match_series(image[it - 1:it + 2, 0:2], END_POINTS, NUM_END_POINTS, 0, 1, 3, 2) == FOUND:
                    found.x = 0
                    found.y = it
                    ends_found.append(found)
                    # return FOUND
    elif top == 1:
        if start == 0:  # top left 2x2
            if image[0, 0] == FIND_VAL and (<int> image[1, 0] + <int> image[1, 1] + <int> image[0, 1]) == FOUND:
                found.x = 0
                found.y = 0
                ends_found.append(found)
                # return FOUND 

        if end == last_col:  # top right 2x2
            if image[0, last_col] == FIND_VAL and (<int> image[0, second_last_col] + <int> image[1, second_last_col] + <int> image[1, last_col]) == FOUND:
                found.x = last_col
                found.y = 0
                ends_found.append(found)
                # return FOUND

        # continue to iterate by 2x3
        if end - 1 >= start + 1:
            for it in range(start + 1, end - 1):
                if image[0, it] == FIND_VAL and convolve_match_series(image[0:2, it - 1:it + 2], END_POINTS, NUM_END_POINTS, 1, 0, 2, 3) == FOUND:
                    found.x = it
                    found.y = 0
                    ends_found.append(found)
                    # return FOUND
    elif right == 1:
        if start == 0:  # top right 2x2
            if image[0, last_col] == FIND_VAL and (<int> image[0, second_last_col] + <int> image[1, second_last_col] + <int> image[1, last_col]) == FOUND:
                found.x = last_col
                found.y = 0
                ends_found.append(found)
                # return FOUND

        if end == last_row:  # bottom right 2x2
            if image[last_row, last_col] == FIND_VAL and (<int> image[last_row, second_last_col] + <int> image[second_last_row, second_last_col] + <int> image[second_last_row, last_col]) == FOUND:
                found.x = last_col
                found.y = last_row
                ends_found.append(found)
                # return FOUND

        # continue to iterate by 2x3
        if end - 1 >= start + 1:
            for it in range(start + 1, end - 1):
                if image[it, last_col] == FIND_VAL and convolve_match_series(image[it - 1:it + 2, second_last_col:cols], END_POINTS, NUM_END_POINTS, 0, 0, 3, 2) == FOUND:
                    found.x = 0
                    found.y = it
                    ends_found.append(found)
                    # return FOUND
    elif bottom == 1:
        if start == 0:  # bottom left 2x2
            if image[last_row, 0] == FIND_VAL and (<int> image[second_last_row, 0] + <int> image[second_last_row, 1] + <int> image[last_row, 1]) == FOUND:
                found.x = 0
                found.y = last_row
                ends_found.append(found)
                # return FOUND

        if end == last_col:  # bottom right 2x2
            if image[last_row, last_col] == FIND_VAL and (<int> image[last_row, second_last_col] + <int> image[second_last_row, second_last_col] + <int> image[second_last_row, last_col]) == FOUND:
                found.x = last_col
                found.y = last_row
                ends_found.append(found)
                # return FOUND

        # continue to iterate by 2x3
        if end - 1 >= start + 1:
            for it in range(start + 1, end - 1):
                if image[last_row, it] == FIND_VAL and convolve_match_series(image[second_last_row:rows, it - 1:it + 2], END_POINTS, NUM_END_POINTS, 0, 0, 2, 3) == FOUND:
                    found.x = it
                    found.y = last_row
                    ends_found.append(found)
                    # return FOUND
    else:
        raise Exception('no edge parameter left,right,top,bottom provided!')

    return ends_found


cpdef np.ndarray[NPINT32_t, ndim=2] old_scan_for_end(NPBOOL_t[:, ::1] image, const Py_ssize_t rows, const Py_ssize_t cols):
    cdef Py_ssize_t start_row
    cdef Py_ssize_t start_col
    cdef Coord found_coord
    cdef int found_edge
    cdef int continue_edge
    cdef list ends_found = []

    # set locs (@TODO add parameter for start scanning location)
    start_row = 0  # start_loc.y
    start_col = 0  # start_loc.x

    # because we're looking at 3x3 matrices we can minimize bulk if statements by scanning the edges first (if applicable)
    if start_row == 0:  # top edge scan
        found_edges = scan_for_edge_end(image, rows, cols, start_col, cols - 1, 0, 0, 1, 0)
        ends_found.extend(found_edges)

        """ @TODO implement start stop scanning for loops
        # stop scanning
        if found_edge:
            return FOUND
        """

        start_row = 1  # we've already scanned top row
        continue_edge = 1  # let's keep scanning in circular pattern
    else:
        continue_edge = 0

    if start_col == cols - 1 or continue_edge == 1:  # right edge scan
        found_edges = scan_for_edge_end(image, rows, cols, start_row, rows - 1, 0, 1, 0, 0)
        ends_found.extend(found_edges)

        """
        # stop scanning
        if found_edge:
            return FOUND
        """

        continue_edge = 1  # let's keep scanning in circular pattern
    
    if start_row == rows - 1 or continue_edge == 1:  # bottom edge scan
        found_edges = scan_for_edge_end(image, rows, cols, start_col, cols - 1, 0, 0, 0, 1)
        ends_found.extend(found_edges)

        """
        # stop scanning
        if found_edges:
            return FOUND
        """

        continue_edge = 1  # let's keep scanning in circular pattern
    
    if start_col == 0 or continue_edge == 1:  # left edge scan
        found_edges = scan_for_edge_end(image, rows, cols, start_row, rows - 1, 1, 0, 0, 0)
        ends_found.extend(found_edges)

        """
        # stop scanning
        if found_edge:
            return FOUND
        """

        start_col = 1  # we've already scanned column 0

    # keep going until we match our first convolved
    cdef Py_ssize_t row, col, width, height, middle_row, middle_col, offset_row, offset_col
    """ @TODO implement the more "effiecent" (as in theoretical) algorithm that loops the image in a circular pattern from the edges
    cdef float relative_col, relative_row
    cdef int direction = 0  # 0 top, 1 right, 2 bottom, 3 left
    # get the width and height of the scanning region (we scan from outer edge to inner edge in a circular pattern)
    middle_row = <Py_ssize_t> int(rows / 2.0)
    middle_col = <Py_ssize_t> int(cols / 2.0)
    
    # determine width offset
    if start_col >= middle_col:
        offset_col = cols - (start_col + 1)
    else:
        offset_col = start_col
    # determine height offset
    if start_row >= middle_row:
        offset_row = rows - (start_row + 1)
    else:
        offset_row = start_row

    # let's get current step in the loop (step being the iteration of inner loop)
    cur_step = 0
    if offset_col >= offset_row and start_row < middle_row:  # more cols completed
        if start_col >= middle_col and offset_col == offset_row:
            direction = 1  # we're in the right corner let's switch direction
        else:
            direction = 0  # top row (for the rest)
        cur_step = offset_row
    elif offset_col > offset_row and start_row >= middle_row:
        if start_col < middle_col and offset_col == offset_row:
            direction = 3  # left column let's switch direction
        else:
            direction = 2  # bottom row
        cur_step = offset_row
    elif offset_row >= offset_col and start_col >= middle_col:
        if start_row >= middle_row and offset_col == offset_row:
            direction = 2  # finished right column let's move left
        else:
            direction = 1  # right col
        cur_step = offset_col
    elif offset_row >= offset_col and start_col < middle_col:
        direction = 3  # left col
        cur_step = offset_col
    else:
        raise Exception('odd loop combination for start position')

    # set current loop width and height
    width = cols - 2 # (cur_step * 2)
    height = rows - 2 # (cur_step * 2)
    cur_col = start_col
    cur_row = start_row

    # for our "starting position" we need to handle a couple special cases of iteration
    # iter_col = range()

    while width >= 0 and height >= 0:
        offset_step = cur_step + 1
        for cur_col in range(cur_col, cols - offset_step):
            if match_end_range(image, cur_row, cur_col, loc) == FOUND:
                return FOUND
        for cur_row in range(cur_row, rows - offset_step):
            if match_end_range(image, cur_row, cur_col, loc) == FOUND:
                return FOUND
        for cur_col in range(cols - offset_step, cur_step, -1):
            if match_end_range(image, cur_row, cur_col, loc) == FOUND:
                return FOUND
        for cur_row in range(rows - offset_step, offset_step, -1):
            if match_end_range(image, cur_row, cur_col, loc) == FOUND:
                return FOUND

    # didn't find anything :(
    loc.x = 0
    loc.y = 0
    return <int> 0
    """

    width = cols - 2
    height = rows - 2

    # edges have been scanned so let's move over to loc 1,1 and continue scanning
    start_row = 1
    start_col = 1

    for row in range(start_row, height):
        for col in range(start_col, width):
            if image[row, col] == FIND_VAL and convolve_match_series(image[row - 1:row + 2, col - 1:col + 2], END_POINTS, NUM_END_POINTS, 0, 0, 3, 3) == FOUND:
                found_coord.x = col
                found_coord.y = row
                ends_found.append(found_coord)

    # convert structs into tuple list
    cdef np.ndarray[NPINT32_t, ndim=2] ret_data = np.zeros((len(ends_found), 2), dtype=np.int32, order='c')
    
    # convert to (row, col) list of matched locations
    for ind, coord in enumerate(ends_found):
        ret_data[ind][0] = coord['y']
        ret_data[ind][1] = coord['x']

    return ret_data
'''

cdef np.ndarray[NPINT32_t, ndim=2] get_c_image_convolve(uint8_t[:, ::1] &image, uint8_t[:, :, ::1] &matches, uint8_t row_first=1, uint8_t scan_edge=0):
    # get results
    cdef vector[location] data
    cdef np.ndarray results
    cdef uint32_t rows, cols, num_matches, match_size, res_size, ind
    cdef location loc
    
    # convert python objects to c
    rows = image.shape[0]
    cols = image.shape[1]
    num_matches = matches.shape[0]
    match_size = matches.shape[1]

    # run c++ optimized image function
    with nogil:
        data = convolve_match_image(&image[0][0], &matches[0][0][0], rows, cols, num_matches, match_size, scan_edge)

    # convert from c++ vector structure into a list of points either (row, col) or (col, row) aka x, y
    res_size = data.size()
    results = np.zeros((res_size, 2), dtype=np.int32, order='c')
    ind = 0
    for loc in data:
        if row_first == <uint8_t> 1:
            results[ind, 0] = loc.row
            results[ind, 1] = loc.col
        else:
            results[ind, 0] = loc.col
            results[ind, 1] = loc.row
        ind += 1

    # return results
    return results


cpdef check_image(uint8_t[:, ::1] &image):
    if image is None:
        raise ValueError('Image cannot be None')

    if image.shape[0] < 5 or image.shape[1] < 5:
        raise ValueError('Image is either empty/too small (fewer than 5 rows/cols) or there are 0 matches to apply to this image')


cpdef np.ndarray[NPINT32_t, ndim=2] get_image_convolve(uint8_t[:, ::1] &image, uint8_t[:, :, ::1] &matches, uint8_t row_first=1, uint8_t scan_edge=0):
    # first check the image
    check_image(image)
    
    # make sure matches are good
    if matches is None:
        raise ValueError('Matches cannot be None')

    # do all of the preliminary checks before we seg fault with/get weird behaviour from our c++ code
    if matches.shape[1] != matches.shape[2] or matches.shape[0] == 0:
        raise ValueError('Matches must be in shape of (match, rows, cols) and the matrix must be a square where rows = cols and cannot be empty')

    # functionality for matrix sizes above 3x3 are iffy
    # if matches.shape[1] != 3 or matches.shape[2] != 3:
    #     raise ValueError('Currently only nx3x3 matching matrices are supported...')

    # we call the wrapper as it's forced to use c, which our loop is optimized for the resulting points
    return get_c_image_convolve(image, matches, row_first, scan_edge)


cpdef np.ndarray[NPINT32_t, ndim=2, mode='c'] scan_for_end(uint8_t[:, ::1] &image, uint8_t row_first=1, uint8_t scan_edge=0):
    # first check the image
    check_image(image)

    # no match checks as we know the matches here are good
    return get_c_image_convolve(image, END_POINTS, row_first, scan_edge)


cpdef np.ndarray[NPINT32_t, ndim=2, mode='c'] scan_for_branch(uint8_t[:, ::1] &image, uint8_t row_first=1, uint8_t scan_edge=0):
    # first check the image
    check_image(image)

    # no match checks as we know the matches here are good
    return get_c_image_convolve(image, BRANCH_POINTS, row_first, scan_edge)


cpdef np.ndarray[NPINT32_t, ndim=3, mode='c'] get_end_point_matches():
    return END_POINTS


cpdef np.ndarray[NPINT32_t, ndim=3, mode='c'] get_branch_point_matches():
    return BRANCH_POINTS


cpdef np.ndarray[NPINT32_t, ndim=2] old_get_end_points(np.ndarray[NPBOOL_t, ndim=2, mode='c'] image):
    if image is None:
        raise ValueError('cannot ')

    cdef np.ndarray[NPINT32_t, ndim=2] endpoints
    cdef Py_ssize_t rows = image.shape[0]
    cdef Py_ssize_t cols = image.shape[1]

    # make sure we're using a C-contiguous array for optimizations
    if image.flags['C_CONTIGUOUS']:
        endpoints = scan_for_end(image, rows, cols)
    else:
        endpoints = scan_for_end(np.ascontiguousarray(image), rows, cols) 
    
    # return list of endpoints
    return endpoints
