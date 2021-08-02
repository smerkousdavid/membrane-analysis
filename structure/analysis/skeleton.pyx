#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
""" Handles skeletonizing blobs. Pulled from scikit-image at https://github.com/scikit-image/scikit-image/blob/70c76fa8a5820e62acd26b0759390f037957978d/skimage/morphology/_skeletonize_cy.pyx """

# numpy
import numpy as np
cimport numpy as np
from structure.analysis.skeleton cimport NPUINT_t
np.import_array()


cpdef np.ndarray[NPUINT_t, ndim=2, mode='c'] fast_skeletonize(NPUINT_t[:, ::1] image):
    """Optimized parts of the Zhang-Suen [1]_ skeletonization.
    Iteratively, pixels meeting removal criteria are removed,
    till only the skeleton remains (that is, no further removable pixel
    was found).
    Performs a hard-coded correlation to assign every neighborhood of 8 a
    unique number, which in turn is used in conjunction with a look up
    table to select the appropriate thinning criteria.
    Parameters
    ----------
    image : numpy.ndarray
        A binary image containing the objects to be skeletonized. '1'
        represents foreground, and '0' represents background.
    Returns
    -------
    skeleton : ndarray
        A matrix containing the thinned image.
    References
    ----------
    .. [1] A fast parallel algorithm for thinning digital patterns,
           T. Y. Zhang and C. Y. Suen, Communications of the ACM,
           March 1984, Volume 27, Number 3.
    """

    # look up table - there is one entry for each of the 2^8=256 possible
    # combinations of 8 binary neighbours. 1's, 2's and 3's are candidates
    # for removal at each iteration of the algorithm.
    cdef int *lut = \
      [0, 0, 0, 1, 0, 0, 1, 3, 0, 0, 3, 1, 1, 0, 1, 3, 0, 0, 0, 0, 0, 0,
       0, 0, 2, 0, 2, 0, 3, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 2, 2, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0,
       0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 2, 0, 0, 0, 3, 1,
       0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 1, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 1, 3, 0, 0,
       1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 2, 3, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3,
       0, 1, 0, 0, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0]

    cdef int pixel_removed, first_pass, neighbors

    # indices for fast iteration
    cdef Py_ssize_t row, col, nrows = image.shape[0]+2, ncols = image.shape[1]+2

    # we copy over the image into a larger version with a single pixel border
    # this removes the need to handle border cases below
    _skeleton = np.zeros((nrows, ncols), dtype=np.uint8, order='c')
    _skeleton[1:nrows-1, 1:ncols-1] = image[:, :] # if it's already bool no need for above 0 > 0
    _cleaned_skeleton = _skeleton.copy(order='c')

    # cdef'd numpy-arrays for fast, typed access
    cdef NPUINT_t[:, ::1] skeleton, cleaned_skeleton

    skeleton = _skeleton
    cleaned_skeleton = _cleaned_skeleton

    pixel_removed = True

    # the algorithm reiterates the thinning till
    # no further thinning occurred (variable pixel_removed set)
    with nogil:
        while pixel_removed:
            pixel_removed = False

            # there are two phases, in the first phase, pixels labeled (see below)
            # 1 and 3 are removed, in the second 2 and 3

            # nogil can't iterate through `(True, False)` because it is a Python
            # tuple. Use the fact that 0 is Falsy, and 1 is truthy in C
            # for the iteration instead.
            # for first_pass in (True, False):
            for pass_num in range(2):
                first_pass = (pass_num == 0)
                for row in range(1, nrows-1):
                    for col in range(1, ncols-1):
                        # all set pixels ...
                        if skeleton[row, col]:
                            # are correlated with a kernel (coefficients spread around here ...)
                            # to apply a unique number to every possible neighborhood ...

                            # which is used with the lut to find the "connectivity type"

                            neighbors = lut[  1*skeleton[row - 1, col - 1] +   2*skeleton[row - 1, col] +\
                                              4*skeleton[row - 1, col + 1] +   8*skeleton[row, col + 1] +\
                                             16*skeleton[row + 1, col + 1] +  32*skeleton[row + 1, col] +\
                                             64*skeleton[row + 1, col - 1] + 128*skeleton[row, col - 1]]

                            # if the condition is met, the pixel is removed (unset)
                            if ((neighbors == 1 and first_pass) or
                                    (neighbors == 2 and not first_pass) or
                                    (neighbors == 3)):
                                cleaned_skeleton[row, col] = 0
                                pixel_removed = True

                # once a step has been processed, the original skeleton
                # is overwritten with the cleaned version
                skeleton[:, :] = cleaned_skeleton[:, :]

    return np.ascontiguousarray(_skeleton[1:nrows-1, 1:ncols-1])

"""
cdef np.ndarray[NPBOOL_t, ndim=2] multi_hit_miss(np.ndarray[NPBOOL_t, ndim=2] image, np.ndarray[NPUINT_t, ndim=3] kernels):
    # buffer image
    cdef int kernel_count = kernels.shape[0]
    cdef int rows = image.shape[0]
    cdef int cols = image.shape[1]
    cdef np.ndarray[NPBOOL_t, ndim=2] buff = np.zeros_like(image)

    # iterate through all of the kernels
    for kernal_image in range(kernel_count):
        hit_or_miss(image, kernels[kernal_image], buff)  # input image, kernel to apply, output image (1 is placed where it matches)
        # buff = np.bitwise_or(hit_or_miss(image, kernels[kernal_image]))
    return buff


def detect_branch_points(input):
    
    hits_misses = multi_hit_miss(input,
                                 )
    hits_misses = np.clip(hits_misses, 0, 1).astype(np.uint8) * 255
    y_hits, x_hits = np.nonzero(hits_misses)
    return [(x_hits[i], y_hits[i]) for i in range(len(x_hits))]


cpdef np.ndarray[NPINT32_t, ndim=2] detect_end_points(np.ndarray[NPBOOL_t, ndim=2] image):
    # X, T and Y kernels
    cdef np.ndarray[NPBOOL_t, ndim=2] hits_misses = multi_hit_miss(image, END_POINTS)
    cdef np.ndarray[NPLONGLONG_t, ndim=1] y_hits, x_hits
    y_hits, x_hits = np.nonzero(hits_misses)
    return np.column_stack((x_hits, y_hits)).astype(np.int32)  # same thing as to the right but more effiecent # [(x_hits[i], y_hits[i]) for i in range(len(x_hits))]
"""