#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

""" Handles skeletonizing blobs. Pulled from scikit-image at https://github.com/scikit-image/scikit-image/blob/70c76fa8a5820e62acd26b0759390f037957978d/skimage/morphology/_skeletonize_cy.pyx """

# numpy
import numpy as np
cimport numpy as np
np.import_array()

# create the type definition 
ctypedef np.uint8_t NPUINT_t
ctypedef unsigned char uint8_t
ctypedef unsigned int uint32_t
ctypedef signed int int32_t
