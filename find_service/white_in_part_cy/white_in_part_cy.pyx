import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def white_in_part_cy(unsigned char[:,:] img, int threshold):
    cdef int i,j,ii,jj
    cdef int ni = img.shape[0]//10
    cdef int nj = img.shape[1]//10
    cdef bint res_ij = False
    cdef int k = 0

    cdef int[:,:] l_sq = np.empty((100,2), dtype='int')

    for i in range(10):
        for j in range(10):
            res_ij = False
            for ii in range(ni):
                if not res_ij:
                    for jj in range(nj):
                        if img[i*ni+ii, j*nj+jj] > threshold:
                            res_ij = True
                            l_sq[k,0]=i
                            l_sq[k,1]=j
                            k = k + 1
                            break
                else:
                    break
    return l_sq[:k]