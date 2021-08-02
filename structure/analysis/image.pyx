from structure.analysis.image cimport get_first_consecutive_row_above_value

""" @TODO finish this module
cpdef np.ndarray[NPINT32_t, ndim=2, mode='c'] autocrop(np.ndarray np.ndarray[NPINT32_t, ndim=2, mode='c'] image) -> np.ndarray:
    "" Automatically crops the specified image to only contain the important/relevant information ""
    if img is None:
        raise ValueError('invalid image provided')
    elif len(img.shape) < 2:
        raise ValueError('invalid image shape of {}'.format(img.shape))
    
    # fix the image depth
    img = fix_depth(img)

    # remove the bottom bar which is usually just a description
    img = remove_bottom_white_box(img)

    # return the fixed image
    return img
"""