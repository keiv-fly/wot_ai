import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport malloc, free
from cython.view cimport array as cvarray

cdef extern from "locs_cy.h":
    ctypedef struct node_t:
        int i
        int j
        node_t* next

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def locs_cy(float[:,:] m_img, double threshold):


    #res = []
    cdef int i,j,k,ni,nk, res_size, i1
    #cdef np.ndarray[np.int, ndim=1] item
    cdef int* res
    cdef int* temp
    cdef int[:,:] res2

    #item = np.empty((2,), dtype = np.int)
    res_size = 100
    res = <int*> malloc(sizeof(int) * res_size *2)
    temp = <int*> malloc(sizeof(int) * res_size *2)
    ni = m_img.shape[0]
    nk = m_img.shape[1]
    #print(m_img.shape)
    #print(ni)
    #print(nk)
    k=0
    with nogil:
        for i in range(ni):
            for j in range(nk):
                if m_img[i,j] < threshold:
                    res[0 + k*2] = j
                    res[1 + k*2] = i
                    k=k+1
                    if k>= res_size:
                        free(temp)
                        temp = res
                        res_size = k*2
                        free(res)
                        res = <int*> malloc(sizeof(int) * k*2 *2)
                        for i1 in range(res_size):
                            res[0 + i1*2] = temp[0 + i1*2]
                            res[1 + i1*2] = temp[1 + i1*2]
                        res_size = k*2


    free(temp)
    if k > 0:
        res2 = <int[:k,:2]> res
        res2 = np.asarray(res2)
    else:
        res2 = np.empty((0,2),dtype=int)
    free(res)

    return res2