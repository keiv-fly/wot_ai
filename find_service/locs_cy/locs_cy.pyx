import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport malloc, free

cdef extern from "locs_cy.h":
    ctypedef struct node_t:
        int i
        int j
        node_t* next

#@cython.boundscheck(False)
#@cython.wraparound(False)
#@cython.nonecheck(False)
def locs_cy(np.ndarray[np.float32_t, ndim=2] m_img, double threshold):


    #res = []
    cdef int i,j,k,ni,nk
    #cdef np.ndarray[np.int, ndim=1] item
    cdef np.ndarray[np.int_t, ndim=2] res

    cdef node_t* l_head = <node_t*>malloc(sizeof(node_t))
    cdef node_t* l_item = l_head
    cdef node_t* l_temp = NULL
    item = np.empty((2,), dtype = np.int)
    ni = m_img.shape[0]
    nk = m_img.shape[1]
    k=0
    with nogil:
        for i in range(ni):
            for j in range(nk):
                if m_img[i,j] < threshold:
                    l_item.i = j
                    l_item.j = i
                    l_temp = <node_t*>malloc(sizeof(node_t))
                    l_item.next = l_temp
                    l_item = l_temp
                    k=k+1
    res = np.empty((k,2), np.int)
    l_item = l_head
    for i in range(k):
        res[i] = (l_item.i, l_item.j)
        l_item = l_item.next

    return res