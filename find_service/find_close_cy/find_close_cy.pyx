import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def remove_close(np.ndarray[np.int32_t, ndim=2] loc):

    cdef np.ndarray[np.int32_t, ndim=2] loc_filtered
    cdef int i,j,n
    cdef np.ndarray[np.int32_t, ndim=1] loc0

    loc_filtered = np.zeros((loc.shape[0],loc.shape[1]), dtype=np.int32)
    n = loc.shape[0]
    j = 0
    for i in range(n):
        loc0 = loc[0]
        loc_filtered[j] = loc0
        j = j + 1
        loc = loc[1:]
        loc = loc[np.abs(np.sum(loc - loc0[np.newaxis, :], axis=1)) > 6]
        if len(loc) == 0:
            break
    loc_filtered = loc_filtered[:j]
    return loc_filtered