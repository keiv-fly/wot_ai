import numpy as np
cimport numpy as np
cimport cython
#from libc.math cimport floor

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def locs_cy(np.ndarray[np.float32_t, ndim=2] m_img, double threshold):

    res = []
    cdef int i,j,ni,nk

    ni = m_img.shape[0]
    nk = m_img.shape[1]
    for i in range(ni):
        for j in range(nk):
            if m_img[i,j] < threshold:
                res.append((j,i))

    return res