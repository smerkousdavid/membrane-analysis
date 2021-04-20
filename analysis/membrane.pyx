""" Handles image masking operations """

# cython
cimport cython
from cpython.mem cimport PyMem_Malloc, PyMem_Free

# numpy
cimport numpy as np
import numpy as np

# custom modules
# cimport skeleton

# python const
NPUINT8 = np.uint8
NPFLOAT = np.float32

# cython const
cdef int dims = 3

# create the type definition
ctypedef np.uint8_t NPUINT_t
ctypedef np.float32_t NPFLOAT_t

# float color structs
cdef struct s_fcolor:
    NPFLOAT_t r, g, b


@cython.boundscheck(False)
@cython.wraparound(False)
cdef blend_mask(int dlen, np.ndarray[NPUINT_t, ndim=3] background, np.ndarray[NPUINT_t, ndim=4] stack, s_fcolor *colors, NPUINT_t min_thresh, NPFLOAT_t alpha):
    """ Blends the background image with a set of binary masks
    
    :note: this will modify the background in place and not make a copy
    :param dlen: the height of the stack
    :param background: the BGR background image (np.uint8) (h, w, 3)
    :param stack: the stack of binary images (np.uint8) (d, h, w, 1)
    :param colors: the struct of colors that matches the depth of the stack
    :param min_thresh: the min class thresh to consider it a valid class identification
    :param alpha: the alpha to apply to each color
    """
    assert background.dtype == NPUINT8
    assert stack.dtype == NPUINT8
    assert sizeof(colors) > 0
    assert dlen == stack.shape[0]
    assert stack.shape[1] == background.shape[0] and stack.shape[2] == background.shape[1]

    # define the width and height and background values
    cdef NPFLOAT_t cur = 0.0
    cdef NPFLOAT_t b_b = 0.0
    cdef NPFLOAT_t b_g = 0.0
    cdef NPFLOAT_t b_r = 0.0
    cdef int width = background.shape[1]
    cdef int height = background.shape[0]
    cdef NPUINT_t[:, :, :] b_view = background
    cdef NPUINT_t[:, :, :, :] s_view = stack

    # loop through each col and row to determine the pixel values
    for row in range(height):
        for col in range(width):
            # get the background color at that index
            b_b = <NPFLOAT_t> b_view[row, col, 0]
            b_g = <NPFLOAT_t> b_view[row, col, 1]
            b_r = <NPFLOAT_t> b_view[row, col, 2]

            # loop through the stack and calculate the new value
            for depth in range(dlen):
                cur = <NPFLOAT_t> s_view[depth, row, col, 0]
                if cur > min_thresh:  # if it passes the min threshold apply the new alpha blend
                    b_b = (alpha * colors[depth].b * cur) + ((1.0 - alpha) * b_b)
                    b_g = (alpha * colors[depth].g * cur) + ((1.0 - alpha) * b_g)
                    b_r = (alpha * colors[depth].r * cur) + ((1.0 - alpha) * b_r)

            # update the final color at that pixel
            b_view[row, col, 0] = <NPUINT_t> b_b
            b_view[row, col, 1] = <NPUINT_t> b_g
            b_view[row, col, 2] = <NPUINT_t> b_r
