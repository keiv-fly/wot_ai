import numpy as np
cimport numpy as np
cimport cython
#from libc.math cimport floor

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def gray_conv(np.ndarray[np.uint8_t, ndim=3] img):

    cdef np.ndarray[np.uint8_t, ndim=2] res
    cdef int i,j,ni,nk

    ni = img.shape[0]
    nk = img.shape[1]
    res = np.empty((ni,nk), dtype=np.uint8)
    for i in range(ni):
        for j in range(nk):
            res[i,j]=int(0.299*img[i,j,2]+0.587*img[i,j,1]+0.114*img[i,j,0])

    return res